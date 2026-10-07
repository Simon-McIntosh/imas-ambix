"""The JT-60SA EDDB batch extractor and its on-demand cache.

The remote process and the EDDB itself are not reachable from a test, so the
transport seam is driven by a fake that answers with synthetic arrays in the
EDDB record shape, and the real :data:`REMOTE_SCRIPT` is executed under the
local python against a fake ``eddb_pwrapper`` module.  Every assertion below is
about the extractor and the cache: one remote process per batch, the ssh prefix
as configuration, the module-loaded remote command, the raw channel landing
with its EDDB attributes and its own time base, the no-op on an already-cached
channel, the refusal to overwrite, the one predicate shared by the reader and
the writer, and the cached store opening through the real engine and view with
no transport call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import zarr

from imas_alambic.eddb import (
    EddbCacheError,
    channel_path,
    channel_time_path,
    eddb_token,
    fetch_channels,
    is_cached,
    normalised_shot,
    read_channel,
    read_eddb_token,
    time_array_name,
    write_channel,
)
from imas_alambic.eddb_remote import (
    DEFAULT_SSH_COMMAND,
    NICE_LEVEL,
    PYTHON_MODULE_LOAD,
    PYTHON_MODULE_UNLOAD,
    REMOTE_SCRIPT,
    BatchResult,
    ChannelRecord,
    ChannelRefusal,
    ChannelRequest,
    EddbRemoteError,
    RemoteEddbExtractor,
    SubprocessTransport,
    decode_batch,
    encode_batch,
    normalise_unit,
)
from imas_alambic.machine_map import ChannelBinding
from imas_alambic.signal_map import MAP_SCHEMA_VERSION, SignalMap, SignalRule
from imas_alambic.transform_engine import ZarrTransformEngine
from imas_alambic.virtual_zarr import VirtualZarrView
from imas_ambix.data.paths import JT60SA_ROOT

# A stand-in for the analysis server's eddb_pwrapper.  It returns a known
# time series for any name except NOTIME, which returns data with no time base,
# and REFUSED, which the EDDB answers with return code 1015, so the per-channel
# refusal path can be exercised without aborting the batch.  The served unit is
# returned as a one-entry list, as the real EDDB wrapper does, so the extractor's
# normalisation to the unit string is exercised end to end.
FAKE_WRAPPER = '''
import numpy as np


class eddbWrapper:
    def __init__(self, lib_path):
        self.lib_path = lib_path

    def eddbOpen(self):
        return True

    def eddbClose(self):
        return True

    def eddbreadOne(self, *args, **kwargs):
        return False, None

    def eddbreadTime(self, shot, category, dname, t1, t2):
        if dname == "NOTIME":
            return True, {"data": np.arange(6.0).reshape(2, 3)}
        if dname == "REFUSED":
            return False, {"irc": 1015, "ircgrp": 1}
        if dname == "ABSENT":
            return False, {"irc": 1013, "ircgrp": 1}
        if dname == "NOIRC":
            return False, {}
        if dname == "MULTIUNIT":
            return True, {
                "data": np.array([[1.0, 2.0, 3.0]]),
                "time": np.array([0.0, 0.5, 1.0]),
                "unit": ["A", "V"],
                "seq": 1,
            }
        return True, {
            "data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "time": np.array([0.0, 0.5, 1.0]),
            "unit": ["A"],
            "seq": 42,
        }
'''


def _request(shot: str, category: str, dname: str) -> ChannelRequest:
    return ChannelRequest(shot=shot, category=category, dname=dname)


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


def _record_from_request(spec: dict[str, str]) -> ChannelRecord:
    return _record(spec["shot"], spec["category"], spec["dname"])


class _FakeTransport:
    """A transport that answers a batch with synthetic EDDB records.

    ``refuse`` maps a data name to the EDDB return code the remote answers it
    with, so a batch can be made to mix served and refused channels.
    """

    def __init__(self, factory=_record_from_request, refuse=None) -> None:
        self.calls: list[tuple[list[str], bytes]] = []
        self._factory = factory
        self._refuse = dict(refuse or {})

    def run(self, argv, payload: bytes) -> bytes:
        self.calls.append((list(argv), payload))
        request_spec = json.loads(payload)
        records = []
        refusals = []
        for spec in request_spec["requests"]:
            code = self._refuse.get(spec["dname"])
            if code is None:
                records.append(self._factory(spec))
            else:
                refusals.append(
                    ChannelRefusal(
                        spec["shot"], spec["category"], spec["dname"], code
                    )
                )
        return encode_batch(records, refusals)


def _extractor(transport=None) -> RemoteEddbExtractor:
    return RemoteEddbExtractor(
        transport=transport if transport is not None else _FakeTransport()
    )


def _binding(category: str, dname: str) -> ChannelBinding:
    return ChannelBinding(
        name="jt60sa-test",
        source_group=category,
        source_array=dname,
        source_rank=2,
        source_role="value",
        source_location="eddb",
        dd_path="magnetics/flux_loop/flux/value",
        source_unit="A",
        target_unit="A",
        sign_convention="identity",
        evidence="synthetic EDDB record for the cache gate",
        source_cocos_override=None,
    )


def _signal_map(category: str, dname: str) -> SignalMap:
    rule = SignalRule(
        semantic_id="cs1",
        source_group=category,
        source_array=dname,
        source_unit="A",
        target_path="magnetics/flux_loop/flux/value",
        target_unit="A",
        target_index=None,
        transformation="one_like",
        source_cocos=None,
        unit_factor=1.0,
        channel_factor=1.0,
        standard_name=None,
        evidence="synthetic EDDB record for the cache gate",
        validation_state="source-only",
    )
    return SignalMap.create(
        schema_version=MAP_SCHEMA_VERSION,
        set_version="0.1.0",
        machine="jt-60sa",
        system="magnetics",
        source_dataset="eddb-live",
        target_dd_version="4.1.1",
        target_cocos=17,
        discovery_producer="imas-codex",
        discovery_receipt="sha256:jt60sa-eddb",
        signals=(rule,),
    )


def _run_remote_script(tmp_path, requests: list[dict[str, str]]):
    """Run the real REMOTE_SCRIPT locally against the fake eddb_pwrapper."""

    (tmp_path / "eddb_pwrapper.py").write_text(FAKE_WRAPPER)
    payload = json.dumps(
        {
            "api_path": str(tmp_path),
            "lib_path": "/analysis/lib/libeddb.so",
            "requests": requests,
        }
    ).encode("utf-8")
    return subprocess.run(
        [sys.executable, "-c", REMOTE_SCRIPT],
        input=payload,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_ssh_command_defaults_to_the_jt60sa_alias_and_is_configurable(tmp_path):
    assert DEFAULT_SSH_COMMAND == ("ssh", "-F", "~/.ssh/config", "jt-60sa")

    default_transport = _FakeTransport()
    fetch_channels(
        _extractor(default_transport),
        tmp_path / "default",
        [_request("1", "MMSYS", "CS1")],
    )
    default_argv, _ = default_transport.calls[0]
    assert default_argv[:4] == [
        "ssh",
        "-F",
        os.path.expanduser("~/.ssh/config"),
        "jt-60sa",
    ]

    custom_transport = _FakeTransport()
    custom = RemoteEddbExtractor(
        ssh_command=("ssh", "-p", "2222", "jt-60sa"), transport=custom_transport
    )
    fetch_channels(custom, tmp_path / "custom", [_request("1", "MMSYS", "CS1")])
    custom_argv, _ = custom_transport.calls[0]
    assert custom_argv[:4] == ["ssh", "-p", "2222", "jt-60sa"]


def test_ssh_config_path_is_expanded_to_an_absolute_path_at_call_time(tmp_path):
    transport = _FakeTransport()
    fetch_channels(_extractor(transport), tmp_path, [_request("1", "MMSYS", "CS1")])
    argv, _ = transport.calls[0]

    assert argv[1] == "-F"
    assert argv[2] == os.path.expanduser("~/.ssh/config")
    assert argv[2].startswith("/")
    assert "~" not in argv[2]


def test_remote_command_loads_the_python_module_and_runs_niced(tmp_path):
    transport = _FakeTransport()
    fetch_channels(_extractor(transport), tmp_path, [_request("1", "MMSYS", "CS1")])
    argv, _ = transport.calls[0]
    shell_command = argv[4]

    assert f"module unload {PYTHON_MODULE_UNLOAD}" in shell_command
    assert f"module load {PYTHON_MODULE_LOAD}" in shell_command
    assert f"nice -n {NICE_LEVEL}" in shell_command
    assert NICE_LEVEL == 19
    assert "python -c" in shell_command


def test_one_transport_process_per_batch(tmp_path):
    transport = _FakeTransport()
    requests = [
        _request("51234", "MMSYS", "CS1"),
        _request("51234", "MMSYS", "EF1"),
        _request("51234", "PSRC", "Ip"),
    ]

    result = fetch_channels(_extractor(transport), tmp_path, requests)

    assert len(transport.calls) == 1
    assert {record.dname for record in result.records} == {"CS1", "EF1", "Ip"}
    assert result.refusals == []


def test_each_channel_lands_raw_with_eddb_attributes_and_its_time_base(tmp_path):
    transport = _FakeTransport()
    expected = _record("51234", "MMSYS", "CS1")

    fetch_channels(_extractor(transport), tmp_path, [_request("51234", "MMSYS", "CS1")])

    path = channel_path(tmp_path, "51234", "MMSYS", "CS1")
    assert path.is_dir()
    stored = zarr.open_array(path, mode="r")
    assert stored[...].shape == expected.data.shape
    assert np.array_equal(stored[...], expected.data)
    assert stored.attrs["units"] == "A"
    assert stored.attrs["channel_count"] == 2
    assert stored.attrs["sequence_number"] == 7

    time_path = channel_time_path(tmp_path, "51234", "MMSYS", "CS1")
    assert time_path.is_dir()
    stored_time = zarr.open_array(time_path, mode="r")
    assert np.array_equal(stored_time[...], expected.time)
    assert stored_time.attrs["units"] == "s"


def test_read_channel_round_trips_data_time_base_and_unit(tmp_path):
    record = _record("51234", "MMSYS", "CS1", nch=2, ntime=5, unit="V", seq=9)
    write_channel(tmp_path, record)

    back = read_channel(tmp_path, "51234", "MMSYS", "CS1")

    assert np.array_equal(back.data, record.data)
    assert np.array_equal(back.time, record.time)
    assert back.unit == "V"
    assert back.nch == 2
    assert back.seq == 9


def test_read_channel_refuses_a_time_base_mismatched_with_the_data(tmp_path):
    write_channel(tmp_path, _record("51234", "MMSYS", "CS1", nch=2, ntime=5))
    group = zarr.open_group(tmp_path / "51234.zarr", mode="a")
    group["MMSYS"].create_array(
        time_array_name("CS1"), data=np.arange(3, dtype="<f8"), overwrite=True
    )

    with pytest.raises(EddbCacheError, match="half-written"):
        read_channel(tmp_path, "51234", "MMSYS", "CS1")


def test_a_cached_channel_causes_no_transport_call(tmp_path):
    transport = _FakeTransport()
    extractor = _extractor(transport)
    first = _request("51234", "MMSYS", "CS1")
    second = _request("51234", "MMSYS", "EF1")

    fetch_channels(extractor, tmp_path, [first])
    assert len(transport.calls) == 1

    fetch_channels(extractor, tmp_path, [first])
    assert len(transport.calls) == 1

    fetch_channels(extractor, tmp_path, [first, second])
    assert len(transport.calls) == 2
    sent = json.loads(transport.calls[1][1])["requests"]
    assert [spec["dname"] for spec in sent] == ["EF1"]


def test_writing_over_an_existing_channel_is_refused(tmp_path):
    record = _record("51234", "MMSYS", "CS1")
    write_channel(tmp_path, record)

    with pytest.raises(EddbCacheError, match="never rewritten"):
        write_channel(tmp_path, replace(record, data=record.data + 1.0))

    stored = zarr.open_array(channel_path(tmp_path, "51234", "MMSYS", "CS1"), mode="r")
    assert np.array_equal(stored[...], record.data)


def test_an_unreadable_channel_node_is_refused_not_reported_not_cached(tmp_path):
    path = channel_path(tmp_path, "51234", "MMSYS", "CS1")
    path.mkdir(parents=True)
    (path / "junk").write_text("not a zarr array")

    with pytest.raises(EddbCacheError, match="not a readable array"):
        is_cached(tmp_path, "51234", "MMSYS", "CS1")

    record = _record("51234", "MMSYS", "CS1")
    with pytest.raises(EddbCacheError, match="not a readable array"):
        write_channel(tmp_path, record)

    transport = _FakeTransport()
    with pytest.raises(EddbCacheError, match="not a readable array"):
        fetch_channels(
            _extractor(transport), tmp_path, [_request("51234", "MMSYS", "CS1")]
        )
    assert transport.calls == []


def test_the_shot_token_is_normalised_to_its_int_form(tmp_path):
    transport = _FakeTransport()
    token = "051234"

    fetch_channels(_extractor(transport), tmp_path, [_request(token, "MMSYS", "CS1")])

    sent = json.loads(transport.calls[0][1])["requests"][0]["shot"]
    assert sent == token

    path = channel_path(tmp_path, token, "MMSYS", "CS1")
    assert "051234" not in str(path)
    assert str(path).endswith("51234.zarr/MMSYS/CS1")
    assert is_cached(tmp_path, token, "MMSYS", "CS1") is True

    with ZarrTransformEngine().open(tmp_path, 51234, "4.1.1") as arrays:
        engine_values = arrays.read(_binding("MMSYS", "CS1"))
    assert engine_values.shape == (2, 5)


def test_series_letter_tokens_land_under_their_integer_and_record_the_token(tmp_path):
    assert normalised_shot("E101173") == "101173"
    assert eddb_token("E101173") == "E101173"
    assert normalised_shot("c510000") == "510000"
    assert eddb_token("c510000") == "C510000"

    transport = _FakeTransport()
    requests = [
        _request("E101173", "MMSYS", "CS1"),
        _request("C510000", "MMSYS", "EF1"),
    ]
    fetch_channels(_extractor(transport), tmp_path, requests)

    sent = [spec["shot"] for spec in json.loads(transport.calls[0][1])["requests"]]
    assert sent == ["E101173", "C510000"]

    path = channel_path(tmp_path, "E101173", "MMSYS", "CS1")
    assert "E101173" not in str(path)
    assert str(path).endswith("101173.zarr/MMSYS/CS1")
    assert read_eddb_token(tmp_path, "E101173") == "E101173"
    assert read_eddb_token(tmp_path, "C510000") == "C510000"
    assert is_cached(tmp_path, "E101173", "MMSYS", "CS1") is True

    with ZarrTransformEngine().open(tmp_path, 101173, "4.1.1") as arrays:
        engine_values = arrays.read(_binding("MMSYS", "CS1"))
    assert engine_values.shape == (2, 5)


def test_a_second_token_sharing_the_digits_is_refused(tmp_path):
    write_channel(tmp_path, _record("E101173", "MMSYS", "CS1"))

    with pytest.raises(EddbCacheError, match="EDDB token"):
        write_channel(tmp_path, _record("C101173", "MMSYS", "CS2"))

    assert is_cached(tmp_path, "E101173", "MMSYS", "CS1") is True
    assert is_cached(tmp_path, "C101173", "MMSYS", "CS2") is False


def test_a_half_written_channel_is_completed_rather_than_refused(tmp_path):
    record = _record("E101173", "MMSYS", "CS1")
    # A failure between the two writes leaves the data array with no time base.
    group = zarr.open_group(tmp_path / "101173.zarr", mode="a")
    group.require_group("MMSYS").create_array(
        "CS1", data=record.data + 50.0, overwrite=False
    )

    assert is_cached(tmp_path, "E101173", "MMSYS", "CS1") is False

    write_channel(tmp_path, record)

    assert is_cached(tmp_path, "E101173", "MMSYS", "CS1") is True
    stored = zarr.open_array(
        channel_path(tmp_path, "E101173", "MMSYS", "CS1"), mode="r"
    )
    assert np.array_equal(stored[...], record.data + 50.0)
    stored_time = zarr.open_array(
        channel_time_path(tmp_path, "E101173", "MMSYS", "CS1"), mode="r"
    )
    assert np.array_equal(stored_time[...], record.time)


def test_a_refetch_completes_a_half_written_channel(tmp_path):
    record = _record("51234", "MMSYS", "CS1")
    group = zarr.open_group(tmp_path / "51234.zarr", mode="a")
    group.require_group("MMSYS").create_array(
        "CS1", data=record.data, overwrite=False
    )

    transport = _FakeTransport()
    fetch_channels(_extractor(transport), tmp_path, [_request("51234", "MMSYS", "CS1")])

    assert len(transport.calls) == 1
    assert is_cached(tmp_path, "51234", "MMSYS", "CS1") is True
    assert channel_time_path(tmp_path, "51234", "MMSYS", "CS1").is_dir()


def test_cached_store_opens_through_engine_and_view_with_no_transport_call(tmp_path):
    transport = _FakeTransport()
    shot = "51234"
    fetch_channels(_extractor(transport), tmp_path, [_request(shot, "MMSYS", "CS1")])
    calls_after_fetch = len(transport.calls)

    with ZarrTransformEngine().open(tmp_path, shot, "4.1.1") as arrays:
        engine_values = arrays.read(_binding("MMSYS", "CS1"))
    assert engine_values.shape == (2, 5)

    store_path = tmp_path / f"{shot}.zarr"
    view = VirtualZarrView.open(
        str(store_path), _signal_map("MMSYS", "CS1"), shot=int(shot)
    )
    assert view.keys() == ("cs1",)
    assert np.array_equal(view["cs1"][:, :3], engine_values[:, :3])

    assert len(transport.calls) == calls_after_fetch


def test_remote_script_executes_and_decodes_a_known_record(tmp_path):
    completed = _run_remote_script(
        tmp_path, [{"shot": "051234", "category": "MMSYS", "dname": "CS1"}]
    )
    assert completed.returncode == 0, completed.stderr.decode()

    result = decode_batch(completed.stdout)
    assert len(result.records) == 1
    assert result.refusals == []
    record = result.records[0]
    assert (record.shot, record.category, record.dname) == ("051234", "MMSYS", "CS1")
    assert np.array_equal(record.data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.array_equal(record.time, [0.0, 0.5, 1.0])
    assert record.unit == "A"
    assert record.nch == 2
    assert record.seq == 42


def test_remote_script_refuses_a_record_with_no_time_base(tmp_path):
    completed = _run_remote_script(
        tmp_path, [{"shot": "051234", "category": "MMSYS", "dname": "NOTIME"}]
    )

    assert completed.returncode != 0
    stderr = completed.stderr.decode()
    assert "no time base" in stderr
    assert "051234" in stderr
    assert "MMSYS" in stderr
    assert "NOTIME" in stderr


def test_remote_script_reports_a_refusal_per_channel_without_aborting_the_batch(
    tmp_path,
):
    completed = _run_remote_script(
        tmp_path,
        [
            {"shot": "051234", "category": "MMSYS", "dname": "CS1"},
            {"shot": "051234", "category": "PSRC", "dname": "REFUSED"},
            {"shot": "051234", "category": "MMSYS", "dname": "EF1"},
        ],
    )
    assert completed.returncode == 0, completed.stderr.decode()

    result = decode_batch(completed.stdout)
    assert [record.dname for record in result.records] == ["CS1", "EF1"]
    assert [refusal.dname for refusal in result.refusals] == ["REFUSED"]
    refusal = result.refusals[0]
    assert (refusal.shot, refusal.category, refusal.dname) == (
        "051234",
        "PSRC",
        "REFUSED",
    )
    assert refusal.code == 1015


def test_remote_script_reads_the_return_code_from_the_wrappers_irc_field(tmp_path):
    # The real wrapper returns (ok, data) with the code in data['irc'] and
    # ircgrp 1 on a refusal; the boolean is the C call status alone, so the code
    # is never a bare number.
    completed = _run_remote_script(
        tmp_path,
        [
            {"shot": "C060033", "category": "PSRC", "dname": "ABSENT"},
            {"shot": "E101173", "category": "PSRC", "dname": "REFUSED"},
        ],
    )
    assert completed.returncode == 0, completed.stderr.decode()

    result = decode_batch(completed.stdout)
    assert result.records == []
    assert {(refusal.dname, refusal.code) for refusal in result.refusals} == {
        ("ABSENT", 1013),
        ("REFUSED", 1015),
    }


def test_remote_script_refuses_a_failed_read_with_no_integer_irc(tmp_path):
    completed = _run_remote_script(
        tmp_path, [{"shot": "E101173", "category": "PSRC", "dname": "NOIRC"}]
    )

    assert completed.returncode != 0
    assert "no integer irc" in completed.stderr.decode()


def test_remote_script_refuses_a_multi_entry_unit(tmp_path):
    completed = _run_remote_script(
        tmp_path, [{"shot": "051234", "category": "MMSYS", "dname": "MULTIUNIT"}]
    )

    assert completed.returncode != 0
    assert "multi-entry unit" in completed.stderr.decode()


def test_fetch_batch_returns_served_records_with_one_refusal_per_refused_channel():
    transport = _FakeTransport(refuse={"Ip": 1015, "magFluxLp1": 1013})
    requests = [
        _request("E101173", "MMSYS", "CS1"),
        _request("E101173", "PSRC", "Ip"),
        _request("E101173", "PSRC", "magFluxLp1"),
    ]

    result = _extractor(transport).fetch_batch(requests)

    assert len(transport.calls) == 1
    assert [record.dname for record in result.records] == ["CS1"]
    assert {(r.category, r.dname, r.code) for r in result.refusals} == {
        ("PSRC", "Ip", 1015),
        ("PSRC", "magFluxLp1", 1013),
    }
    assert {r.shot for r in result.refusals} == {"E101173"}


def test_fetch_batch_makes_no_call_for_an_empty_batch():
    transport = _FakeTransport()

    result = _extractor(transport).fetch_batch([])

    assert result == BatchResult(records=[], refusals=[])
    assert transport.calls == []


def test_fetch_batch_raises_only_when_the_session_itself_fails():
    class _BrokenTransport:
        def run(self, argv, payload):
            raise EddbRemoteError("ssh: connect to host jt-60sa port 22: refused")

    with pytest.raises(EddbRemoteError, match="connect to host"):
        _extractor(_BrokenTransport()).fetch_batch([_request("1", "MMSYS", "CS1")])

    # A refused channel is reported in the result, not raised.
    transport = _FakeTransport(refuse={"CS1": 1015})
    result = _extractor(transport).fetch_batch([_request("1", "MMSYS", "CS1")])
    assert result.records == []
    assert result.refusals[0].code == 1015


def test_fetch_channels_caches_the_served_and_reports_the_refused(tmp_path):
    transport = _FakeTransport(refuse={"Ip": 1015})
    requests = [
        _request("E101173", "MMSYS", "CS1"),
        _request("E101173", "PSRC", "Ip"),
    ]

    result = fetch_channels(_extractor(transport), tmp_path, requests)

    assert [record.dname for record in result.records] == ["CS1"]
    assert [refusal.dname for refusal in result.refusals] == ["Ip"]
    assert result.refusals[0].code == 1015
    assert is_cached(tmp_path, "E101173", "MMSYS", "CS1") is True
    assert is_cached(tmp_path, "E101173", "PSRC", "Ip") is False


def test_a_batch_that_is_entirely_refused_writes_nothing(tmp_path):
    transport = _FakeTransport(refuse={"Ip": 1015, "CS1": 1013})
    requests = [
        _request("E101173", "PSRC", "Ip"),
        _request("E101173", "MMSYS", "CS1"),
    ]

    result = fetch_channels(_extractor(transport), tmp_path, requests)

    assert result.records == []
    assert {refusal.dname for refusal in result.refusals} == {"Ip", "CS1"}
    assert list(tmp_path.iterdir()) == []


def test_the_cached_unit_is_the_eddb_unit_string_not_its_repr(tmp_path):
    completed = _run_remote_script(
        tmp_path, [{"shot": "051234", "category": "MMSYS", "dname": "CS1"}]
    )
    record = decode_batch(completed.stdout).records[0]
    assert record.unit == "A"

    cache_root = tmp_path / "cache"
    write_channel(cache_root, record)
    stored = zarr.open_array(
        channel_path(cache_root, "051234", "MMSYS", "CS1"), mode="r"
    )
    assert stored.attrs["units"] == "A"


def test_normalise_unit_takes_a_single_entry_and_refuses_a_multi_entry_list():
    assert normalise_unit(["A"]) == "A"
    assert normalise_unit("A") == "A"
    assert normalise_unit(None) == ""

    with pytest.raises(EddbRemoteError, match="multi-entry unit"):
        normalise_unit(["A", "V"])


def test_envelope_round_trips_through_encode_and_decode():
    records = [_record("51234", "MMSYS", "CS1"), _record("51234", "PSRC", "Ip")]
    decoded = decode_batch(encode_batch(records))
    assert [record.key for record in decoded.records] == [
        record.key for record in records
    ]
    assert decoded.refusals == []
    assert np.array_equal(decoded.records[1].data, records[1].data)
    assert decoded.records[0].unit == "A"


def test_default_transport_is_the_only_subprocess_seam():
    assert isinstance(RemoteEddbExtractor().transport, SubprocessTransport)


def test_jt60sa_cache_root_is_declared_once():
    assert str(JT60SA_ROOT) == "/work/projects/imas_gpu/jt60sa"
