"""The JT-60SA EDDB batch extractor and its on-demand cache.

The remote process and the EDDB itself are not reachable from a test, so the
transport seam is driven by a fake that answers with synthetic arrays in the
EDDB record shape.  Every assertion below is about the extractor and the cache:
one remote process per batch, the ssh prefix as configuration, the raw channel
landing with its EDDB attributes, the no-op on an already-cached channel, the
refusal to overwrite, and the cached store opening through the real engine and
view with no transport call.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import zarr

from imas_ambix.data.eddb import (
    EddbCacheError,
    channel_path,
    fetch_channels,
    write_channel,
)
from imas_ambix.data.eddb_remote import (
    DEFAULT_SSH_COMMAND,
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


def test_ssh_command_is_configuration_and_defaults_to_the_jt60sa_alias(tmp_path):
    assert DEFAULT_SSH_COMMAND == ("ssh", "-F", "~/.ssh/config", "jt-60sa")

    default_transport = _FakeTransport()
    default_extractor = RemoteEddbExtractor(transport=default_transport)
    fetch_channels(
        default_extractor, tmp_path / "default", [_request("1", "MMSYS", "CS1")]
    )
    default_argv, _ = default_transport.calls[0]
    assert default_argv[: len(DEFAULT_SSH_COMMAND)] == list(DEFAULT_SSH_COMMAND)

    custom_transport = _FakeTransport()
    custom = RemoteEddbExtractor(
        ssh_command=("ssh", "-p", "2222", "jt-60sa"), transport=custom_transport
    )
    fetch_channels(custom, tmp_path / "custom", [_request("1", "MMSYS", "CS1")])
    custom_argv, _ = custom_transport.calls[0]
    assert custom_argv[:4] == ["ssh", "-p", "2222", "jt-60sa"]


def test_one_transport_process_per_batch(tmp_path):
    transport = _FakeTransport()
    extractor = RemoteEddbExtractor(transport=transport)
    requests = [
        _request("51234", "MMSYS", "CS1"),
        _request("51234", "MMSYS", "EF1"),
        _request("51234", "PSRC", "Ip"),
    ]

    records = fetch_channels(extractor, tmp_path, requests)

    assert len(transport.calls) == 1
    assert {record.dname for record in records} == {"CS1", "EF1", "Ip"}


def test_each_channel_lands_raw_with_eddb_attributes(tmp_path):
    transport = _FakeTransport()
    extractor = RemoteEddbExtractor(transport=transport)
    request = _request("51234", "MMSYS", "CS1")
    expected = _record("51234", "MMSYS", "CS1")

    fetch_channels(extractor, tmp_path, [request])

    path = channel_path(tmp_path, "51234", "MMSYS", "CS1")
    assert path.is_dir()
    stored = zarr.open_array(path, mode="r")
    assert stored[...].shape == expected.data.shape
    assert np.array_equal(stored[...], expected.data)
    assert stored.attrs["units"] == "A"
    assert stored.attrs["channel_count"] == 2
    assert stored.attrs["sequence_number"] == 7


def test_a_cached_channel_causes_no_transport_call(tmp_path):
    transport = _FakeTransport()
    extractor = RemoteEddbExtractor(transport=transport)
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


def test_cached_store_opens_through_engine_and_view_with_no_transport_call(tmp_path):
    transport = _FakeTransport()
    extractor = RemoteEddbExtractor(transport=transport)
    shot = "51234"
    fetch_channels(extractor, tmp_path, [_request(shot, "MMSYS", "CS1")])
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


def test_the_remote_script_uses_stdlib_numpy_and_the_wrapper_only():
    from imas_ambix.data.eddb_remote import REMOTE_SCRIPT

    assert "eddb_pwrapper" in REMOTE_SCRIPT
    assert "eddbreadTime" in REMOTE_SCRIPT
    assert "numpy" in REMOTE_SCRIPT
    assert "import requests" not in REMOTE_SCRIPT


def test_envelope_round_trips_through_encode_and_decode():
    records = [_record("51234", "MMSYS", "CS1"), _record("51234", "PSRC", "Ip")]
    decoded = decode_batch(encode_batch(records))
    assert [record.key for record in decoded] == [record.key for record in records]
    assert np.array_equal(decoded[1].data, records[1].data)
    assert decoded[0].unit == "A"


def test_default_ssh_transport_is_the_only_network_seam():
    extractor = RemoteEddbExtractor()
    assert isinstance(extractor.transport, SshTransport)


def test_jt60sa_cache_root_is_declared_once():
    assert str(JT60SA_ROOT) == "/work/projects/imas_gpu/jt60sa"
