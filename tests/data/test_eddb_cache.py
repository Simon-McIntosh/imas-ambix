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

from imas_ambix.data.eddb import (
    EddbCacheError,
    channel_path,
    channel_time_path,
    fetch_channels,
    is_cached,
    write_channel,
)
from imas_ambix.data.eddb_remote import (
    DEFAULT_SSH_COMMAND,
    NICE_LEVEL,
    PYTHON_MODULE_LOAD,
    PYTHON_MODULE_UNLOAD,
    REMOTE_SCRIPT,
    ChannelRecord,
    ChannelRequest,
    RemoteEddbExtractor,
    SshTransport,
    decode_batch,
    encode_batch,
)
from imas_ambix.data.machine_map import ChannelBinding
from imas_ambix.data.paths import JT60SA_ROOT
from imas_ambix.data.signal_map import MAP_SCHEMA_VERSION, SignalMap, SignalRule
from imas_ambix.data.transform_engine import ZarrTransformEngine
from imas_ambix.data.virtual_zarr import VirtualZarrView

# A stand-in for the analysis server's eddb_pwrapper.  It returns a known
# time series for any name except NOTIME, which returns data with no time base
# so the refusal path can be exercised.
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
        return True, {
            "data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "time": np.array([0.0, 0.5, 1.0]),
            "unit": "A",
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
    """A transport that answers a batch with synthetic EDDB records."""

    def __init__(self, factory=_record_from_request) -> None:
        self.calls: list[tuple[list[str], bytes]] = []
        self._factory = factory

    def run(self, argv, payload: bytes) -> bytes:
        self.calls.append((list(argv), payload))
        request_spec = json.loads(payload)
        return encode_batch(
            [self._factory(spec) for spec in request_spec["requests"]]
        )


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

    records = fetch_channels(_extractor(transport), tmp_path, requests)

    assert len(transport.calls) == 1
    assert {record.dname for record in records} == {"CS1", "EF1", "Ip"}


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


def test_a_present_channel_is_cached_and_refused_even_when_not_an_array(tmp_path):
    path = channel_path(tmp_path, "51234", "MMSYS", "CS1")
    path.mkdir(parents=True)
    (path / "junk").write_text("not a zarr array")

    assert is_cached(tmp_path, "51234", "MMSYS", "CS1") is True

    record = _record("51234", "MMSYS", "CS1")
    with pytest.raises(EddbCacheError):
        write_channel(tmp_path, record)

    transport = _FakeTransport()
    fetch_channels(_extractor(transport), tmp_path, [_request("51234", "MMSYS", "CS1")])
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

    records = decode_batch(completed.stdout)
    assert len(records) == 1
    record = records[0]
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


def test_envelope_round_trips_through_encode_and_decode():
    records = [_record("51234", "MMSYS", "CS1"), _record("51234", "PSRC", "Ip")]
    decoded = decode_batch(encode_batch(records))
    assert [record.key for record in decoded] == [record.key for record in records]
    assert np.array_equal(decoded[1].data, records[1].data)
    assert decoded[0].unit == "A"


def test_default_ssh_transport_is_the_only_network_seam():
    assert isinstance(RemoteEddbExtractor().transport, SshTransport)


def test_jt60sa_cache_root_is_declared_once():
    assert str(JT60SA_ROOT) == "/work/projects/imas_gpu/jt60sa"
