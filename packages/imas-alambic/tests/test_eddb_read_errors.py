"""``read_channel`` names the cache fault for a bare-digit pulse.

The cache addresses a pulse by the integer of its digits, so a bare integer
names a valid cache directory.  A read that misses the cache, or finds a
half-written channel, names that cache fault rather than the remote address
grammar: only :func:`eddb_token`, the address that leaves for EDDB, demands the
series letter.  The tests pin the bare-digit message for both faults, the
unchanged lettered message, and that the strict token still refuses bare digits.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from imas_alambic.eddb import (
    EddbCacheError,
    eddb_token,
    read_channel,
    time_array_name,
    write_channel,
)
from imas_alambic.eddb_remote import ChannelRecord


def _record(
    shot: str,
    category: str,
    dname: str,
    *,
    nch: int = 2,
    ntime: int = 5,
    unit: str = "A",
    seq: int = 7,
) -> ChannelRecord:
    data = np.arange(nch * ntime, dtype="<f8").reshape(nch, ntime) + 100.0
    time = np.arange(ntime, dtype="<f8") * 0.1
    return ChannelRecord(shot, category, dname, data, time, unit, nch, seq)


def test_a_bare_digit_miss_names_the_cache_fault_not_the_token_refusal(tmp_path):
    with pytest.raises(EddbCacheError) as raised:
        read_channel(tmp_path, 101154, "MMSYS", "CS1")

    message = str(raised.value)
    assert "101154/MMSYS/CS1 is not cached at" in message
    assert "is not an EDDB pulse token" not in message


def test_a_bare_digit_half_written_channel_names_the_half_written_fault(tmp_path):
    write_channel(tmp_path, _record("E101154", "MMSYS", "CS1"))
    group = zarr.open_group(tmp_path / "101154.zarr", mode="a")
    group["MMSYS"].create_array(
        time_array_name("CS1"), data=np.arange(3, dtype="<f8"), overwrite=True
    )

    with pytest.raises(EddbCacheError) as raised:
        read_channel(tmp_path, 101154, "MMSYS", "CS1")

    message = str(raised.value)
    assert "101154/MMSYS/CS1 is half-written" in message
    assert "is not an EDDB pulse token" not in message


def test_a_lettered_token_reads_the_not_cached_message_exactly_as_before(tmp_path):
    with pytest.raises(EddbCacheError) as raised:
        read_channel(tmp_path, "E101154", "MMSYS", "CS1")

    assert "channel E101154/MMSYS/CS1 is not cached at" in str(raised.value)


def test_the_strict_token_still_refuses_a_bare_digit():
    with pytest.raises(EddbCacheError, match="series letter"):
        eddb_token(101154)
