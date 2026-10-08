from __future__ import annotations

import os
from math import tau
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from imas_alambic.cocos import CANONICAL_COCOS
from imas_ambix.challenge.convention import DIIID_CONVENTION
from imas_ambix.challenge.facts import build_report
from imas_ambix.challenge.loader import load_geqdsk, load_shot, validate_shot_schema


def _real_slice() -> list[Path]:
    root = Path(
        os.environ.get(
            "SOPHELIO_DIIID_TRAIN",
            "/work/projects/imas_gpu/sophelio/raw/data/diii_d_train",
        )
    )
    return sorted(root.glob("*.parquet"))[:100]


# The hundred-shot DIII-D schema sweep measured 753 s on 2026-10-05
# (nproc 28, load1 10); the bound is twice that rounded up to a whole minute
# so the corpus read is not cut off by the 300 s default.
@pytest.mark.timeout(1560)
def test_real_diii_d_schema_on_one_hundred_shots() -> None:
    paths = _real_slice()
    if len(paths) < 100:
        pytest.skip(f"real corpus slice has {len(paths)} of 100 required shots")
    validated = 0
    for path in paths:
        shot = validate_shot_schema(path)
        assert shot.source == "DIII-D"
        assert shot.labels.psirz.shape[1:] == (65, 65)
        assert shot.labels.cocos == CANONICAL_COCOS
        validated += 1
    assert validated == 100


# The facts-report build over the same hundred-shot slice measured 724 s on
# 2026-10-05 (nproc 28, load1 10); the bound is twice that rounded up to a
# whole minute so the corpus read is not cut off by the 300 s default.
@pytest.mark.timeout(1500)
def test_facts_report_covers_every_circulated_claim() -> None:
    paths = _real_slice()
    if len(paths) < 100:
        pytest.skip(f"real corpus slice has {len(paths)} of 100 required shots")
    report = build_report(paths)
    claims = report["claims"]
    assert len(claims) == 5
    assert {claim["verdict"] for claim in claims} <= {
        "confirmed",
        "corrected",
        "unreachable-from-slice",
    }
    assert report["measurements"]["shots"] == 100


def _write_small_geqdsk(path: Path, *, psi_sign: float = 1.0) -> Path:
    """Write one small G-EQDSK under a temporary path for the loader to read."""

    from eqdsk import EQDSKInterface

    size = 9
    radius = np.linspace(2.0, 4.0, size)
    height = np.linspace(-1.5, 1.5, size)
    flux = psi_sign * np.transpose(
        (radius[np.newaxis, :] - 3.0) ** 2 + height[:, np.newaxis] ** 2
    )
    angle = np.linspace(0.0, tau, 16, endpoint=False)
    instance = EQDSKInterface(
        bcentre=2.5,
        cplasma=8.0e5,
        dxc=np.zeros(0),
        dzc=np.zeros(0),
        ffprime=np.zeros(size),
        fpol=np.full(size, 6.0),
        Ic=np.zeros(0),
        name="small",
        nbdry=angle.size,
        ncoil=0,
        nlim=0,
        nx=size,
        nz=size,
        pprime=np.zeros(size),
        pressure=np.zeros(size),
        psi=flux,
        psibdry=float(flux[0, 0]),
        psimag=float(flux[size // 2, size // 2]),
        xbdry=3.0 + 0.6 * np.cos(angle),
        xc=np.zeros(0),
        xcentre=3.0,
        xdim=2.0,
        xgrid1=2.0,
        xlim=np.zeros(0),
        xmag=3.0,
        zbdry=0.6 * np.sin(angle),
        zc=np.zeros(0),
        zdim=3.0,
        zlim=np.zeros(0),
        zmag=0.0,
        zmid=0.0,
        qpsi=np.linspace(2.0, 5.0, size),
    )
    instance.write(path, file_format="geqdsk")
    return path


def test_geqdsk_loader_reads_a_file_into_efit_labels(tmp_path: Path) -> None:
    """A G-EQDSK reads into the typed record the convention audit consumes."""

    path = _write_small_geqdsk(tmp_path / "small.geqdsk")

    record = load_geqdsk(path, time_ms=2.5)

    assert record.time_ms == pytest.approx([2.5])
    assert record.psirz.shape == (1, 9, 9)
    assert record.grid_r_m.shape == (9,)
    assert record.grid_z_m.shape == (9,)
    assert record.lcfs_r_m.shape == (1, 16)
    assert record.cocos == 1
    assert set(record.scalars) == {
        "efit_q95",
        "efit_r_axis",
        "efit_z_axis",
        "magnetics_bcoil",
    }
    assert record.scalars["efit_q95"][0] == pytest.approx(4.85)
    assert record.scalars["efit_r_axis"][0] == pytest.approx(3.0)
    assert record.scalars["magnetics_bcoil"][0] == pytest.approx(2.5)
    assert record.psirz[0, 0, 0] - record.psirz[0, 4, 4] > 0.0


def test_geqdsk_loader_keeps_the_files_psi_sign(tmp_path: Path) -> None:
    """The record carries the file's own psi, not an assumed sign."""

    positive = load_geqdsk(
        _write_small_geqdsk(tmp_path / "positive.geqdsk"), time_ms=2.5
    )
    negative = load_geqdsk(
        _write_small_geqdsk(tmp_path / "negative.geqdsk", psi_sign=-1.0), time_ms=2.5
    )

    assert positive.psirz[0, 0, 0] - positive.psirz[0, 4, 4] > 0.0
    assert negative.psirz[0, 0, 0] - negative.psirz[0, 4, 4] < 0.0


def test_loader_serves_diii_d_labels_in_the_canonical_convention() -> None:
    paths = _real_slice()
    if not paths:
        pytest.skip("real DIII-D corpus is unavailable")
    path = paths[0]
    raw = pq.read_table(
        path,
        columns=[
            "efit_psirz",
            "efit_q95",
            "magnetics_plasma_current",
            "magnetics_bcoil",
        ],
    )
    shot = load_shot(path)

    raw_flux = np.asarray(raw["efit_psirz"][0].as_py(), dtype=float)
    raw_q95 = np.asarray(raw["efit_q95"][0].as_py(), dtype=float)
    raw_ip = np.asarray(raw["magnetics_plasma_current"][0].as_py(), dtype=float)
    raw_bcoil = np.asarray(raw["magnetics_bcoil"][0].as_py(), dtype=float)

    np.testing.assert_allclose(shot.labels.psirz, -tau * raw_flux)
    np.testing.assert_allclose(shot.labels.scalars["efit_q95"], -raw_q95)
    np.testing.assert_allclose(
        shot.actuators["magnetics_plasma_current"].values,
        DIIID_CONVENTION.canonical_plasma_current(raw_ip),
    )
    np.testing.assert_allclose(
        shot.actuators["magnetics_bcoil"].values,
        DIIID_CONVENTION.canonical_toroidal_field(raw_bcoil),
    )
