"""Compressed engine-metrics responses retain the lane reader contract."""

from __future__ import annotations

import gzip
from pathlib import Path

from imas_ambix.agent import lane

_FIXTURE = Path(__file__).parent / "data" / "sglang_metrics_sample.txt"


class _CompressedResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = {"Content-Encoding": "gzip"}

    def __enter__(self) -> _CompressedResponse:
        return self

    def __exit__(self, *unused: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _CompressedOpener:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def open(self, request: object, timeout: float) -> _CompressedResponse:
        return _CompressedResponse(self._body)


def test_gzip_metrics_match_the_equivalent_plain_capacity(monkeypatch):
    """A declared gzip body parses identically to the checked-in receipt."""
    plain = _FIXTURE.read_text(encoding="utf-8")
    expected = lane.parse_lane_capacity(plain)
    monkeypatch.setattr(
        lane.urllib.request,
        "build_opener",
        lambda handler: _CompressedOpener(gzip.compress(plain.encode("utf-8"))),
    )

    assert lane.fetch_lane_capacity("http://fixture.invalid") == expected
