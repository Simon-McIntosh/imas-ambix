"""The local EDDB reader preserves cached facility arrays bit for bit."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import zarr

from imas_alambic.eddb_remote import ChannelRequest
from imas_ambix.data.paths import JT60SA_ROOT
from tests.jt60sa_bundle import BUNDLE, SKIP_REASON

# The engine's own test package owns the EDDB stand-in wrapper and the helpers
# that place it: the engine tests must run in a clean environment holding only
# the engine wheels and pytest, so that owner is not a package this suite can
# import by name.  Load it by file path and take the names from the loaded
# module, so no directory is placed on ``sys.path``.
_ENGINE_STANDIN = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "imas-alambic"
    / "tests"
    / "eddb_standin.py"
)


def _load_engine_standin():
    spec = importlib.util.spec_from_file_location("eddb_standin", _ENGINE_STANDIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_eddb_standin = _load_engine_standin()
local_extractor = _eddb_standin.local_extractor
write_standin = _eddb_standin.write_standin

LOGGER = logging.getLogger("test_eddb_local_transport")
PF_ACTIVE_ARRAYS = (
    "curCS1LKAT",
    "curCS2LKAT",
    "curCS3LKAT",
    "curCS4LKAT",
    "curEF1LKAT",
    "curEF2LKAT",
    "curEF3LKAT",
    "curEF4LKAT",
    "curEF5LKAT",
    "curEF6HiTe",
)
PF_ACTIVE_CATEGORY = "MMSYS"
PF_ACTIVE_SHOT = "E101154"

def _request(dname: str) -> ChannelRequest:
    return ChannelRequest(shot=PF_ACTIVE_SHOT, category=PF_ACTIVE_CATEGORY, dname=dname)


def test_local_route_matches_cached_pf_active_arrays_bit_for_bit(tmp_path):
    if BUNDLE is None:
        pytest.skip(SKIP_REASON)
    store = JT60SA_ROOT / "101154.zarr"
    if not store.is_dir():
        pytest.skip(f"cached pulse {store} is not present")

    channels = {}
    for dname in PF_ACTIVE_ARRAYS:
        data = np.asarray(
            zarr.open_array(store / PF_ACTIVE_CATEGORY / dname, mode="r")[...]
        )
        time = np.asarray(
            zarr.open_array(store / PF_ACTIVE_CATEGORY / f"{dname}_time", mode="r")[...]
        )
        channels[dname] = (data, time, 0)
    write_standin(tmp_path, channels)

    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Collect()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        result = local_extractor(tmp_path).fetch_batch(
            [_request(d) for d in PF_ACTIVE_ARRAYS]
        )
        assert result.refusals == []
        assert len(result.records) == len(PF_ACTIVE_ARRAYS)
        for record in result.records:
            standin_data, standin_time, _ = channels[record.dname]
            assert record.data.shape == standin_data.shape
            assert record.time.shape == standin_time.shape
            assert np.array_equal(record.data, standin_data)
            assert np.array_equal(record.time, standin_time)
            LOGGER.info(
                "local pf_active %s data%s time%s equal stand-in arrays bit for bit",
                record.dname,
                record.data.shape,
                record.time.shape,
            )
    finally:
        LOGGER.removeHandler(handler)

    assert sum("equal stand-in arrays bit for bit" in m for m in messages) == len(
        PF_ACTIVE_ARRAYS
    )
