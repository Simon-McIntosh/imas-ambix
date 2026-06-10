"""Shared fixtures for the camdyn test suite.

A synthetic V3 token store + V2 level-1 store are built in a tmp dir so
the tests are CPU-fast and do not depend on the 9,527-shot corpus being
mounted.  A small real-shot smoke (24065) runs only when the corpus path
exists (skipped otherwise).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REAL_TOKEN_SHOT = 24065
REAL_TOKEN_PATH = Path(
    f"/work/projects/imas_gpu/mast-tokens/v1/frames/{REAL_TOKEN_SHOT}/rbb.zarr"
)
REAL_LEVEL1_PATH = Path(
    f"/work/projects/imas_gpu/mast/level1/shots/{REAL_TOKEN_SHOT}.zarr"
)


@pytest.fixture
def synthetic_corpus(tmp_path: Path):
    """Build a tiny synthetic token + level-1 corpus.

    Returns a dict with ``token_root`` / ``level1_dir`` / ``shot_ids`` /
    ``n_frames`` for use with the camdyn dataset + conditioning loaders.
    """
    import zarr

    vocab_version = "v1"
    camera = "rbb"
    shot_ids = [90001, 90002, 90003]
    n_frames = {90001: 40, 90002: 25, 90003: 8}  # one short shot

    token_root = tmp_path / "tokens"
    level1_dir = tmp_path / "level1"
    frames_dir = token_root / vocab_version / "frames"
    frames_dir.mkdir(parents=True)
    level1_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    dt = 1.0 / 600.0
    for sid in shot_ids:
        nf = n_frames[sid]
        # --- V3 token store ---
        tpath = frames_dir / str(sid) / f"{camera}.zarr"
        tstore = zarr.open_group(str(tpath), mode="w")
        toks = rng.integers(0, 1 << 18, size=(nf, 16, 16), dtype=np.int32)
        tstore.create_array("tokens", shape=toks.shape, dtype=toks.dtype)
        tstore["tokens"][:] = toks
        tstore.attrs["shot_id"] = sid
        tstore.attrs["camera"] = camera

        # --- V2 level-1 store with rbb/time + a few actuators ---
        lpath = level1_dir / f"{sid}.zarr"
        lstore = zarr.open_group(str(lpath), mode="w")
        rbb = lstore.create_group("rbb")
        ft = 0.01 + dt * np.arange(nf, dtype=np.float64)
        rbb.create_array("time", shape=ft.shape, dtype=ft.dtype)
        rbb["time"][:] = ft
        rbb.create_array("data", shape=(nf, 4, 4), dtype="uint8")  # dummy raw

        # amc actuators on a finer grid spanning the frame window
        amc = lstore.create_group("amc")
        at = np.linspace(ft[0] - 0.05, ft[-1] + 0.05, 200).astype(np.float64)
        amc.create_array("time", shape=at.shape, dtype=at.dtype)
        amc["time"][:] = at
        for name, val in [
            ("plasma_current", 400.0 + 50.0 * np.sin(at * 10)),
            ("p4u_coil_current", 1.2 + 0.1 * np.cos(at * 5)),
            ("sol_current", -10.0 + at),
            ("tf_current", 100.0 + 0 * at),
        ]:
            amc.create_array(name, shape=val.shape, dtype="float32")
            amc[name][:] = val.astype(np.float32)

        # anb beam power (later start → some leading-missing frames)
        anb = lstore.create_group("anb")
        bt = np.linspace(ft[5] if nf > 5 else ft[-1], ft[-1] + 0.02, 80)
        anb.create_array("time", shape=bt.shape, dtype="float64")
        anb["time"][:] = bt.astype(np.float64)
        for name in ("ss_sum_power", "sw_sum_power", "tot_sum_power"):
            v = 0.5 + 0.0 * bt
            anb.create_array(name, shape=v.shape, dtype="float32")
            anb[name][:] = v.astype(np.float32)

        # aga gas puff
        aga = lstore.create_group("aga")
        gt = np.linspace(ft[0] - 0.02, ft[-1] + 0.02, 300)
        aga.create_array("time", shape=gt.shape, dtype="float64")
        aga["time"][:] = gt.astype(np.float64)
        for name in (
            "inboard_total",
            "inboard_upper",
            "inboard_lower",
            "outboard_total",
        ):
            v = 1e19 + 1e18 * np.sin(gt * 3)
            aga.create_array(name, shape=v.shape, dtype="float32")
            aga[name][:] = v.astype(np.float32)

        # ane line-integrated density
        ane = lstore.create_group("ane")
        net = np.linspace(ft[0], ft[-1], 150)
        ane.create_array("time", shape=net.shape, dtype="float64")
        ane["time"][:] = net.astype(np.float64)
        nev = 1.0e19 + 5e18 * np.sin(net * 2)
        ane.create_array("density", shape=nev.shape, dtype="float32")
        ane["density"][:] = nev.astype(np.float32)

        # ada Dα (probe target)
        ada = lstore.create_group("ada")
        adt = np.linspace(ft[0], ft[-1], 50)
        ada.create_array("time", shape=adt.shape, dtype="float64")
        ada["time"][:] = adt.astype(np.float64)
        adv = 2.0 + np.cos(adt * 4)
        ada.create_array("dalpha_integrated", shape=adv.shape, dtype="float32")
        ada["dalpha_integrated"][:] = adv.astype(np.float32)

    return {
        "token_root": token_root,
        "level1_dir": level1_dir,
        "shot_ids": shot_ids,
        "n_frames": n_frames,
        "vocab_version": vocab_version,
        "camera": camera,
    }
