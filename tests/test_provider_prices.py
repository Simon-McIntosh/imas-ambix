"""One module owns the provider price table: fetch, write, age it.

Every test here stubs the network fetch, so nothing binds a socket. The table
is the one document the whole ledger's cost column is priced from, and the
clauses fixed here are the ones a second owner would break: the path resolves
through :class:`SiteConfig`; the age comes from the document's own ``at``
field rather than a copy or restore's mtime; a table younger than the refresh
window is not refetched; a fetch that fails or returns nothing leaves the
cached table in force and reports the age of what was read; the write is a
scratch file renamed onto the path; and the retired cache seeds the new path
once and never again.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.request
from pathlib import Path

import pytest

from imas_ambix.agent import provider_prices, watch
from imas_ambix.agent.profile import SiteConfig

DAY = 24 * 3600.0
NOW = 1_800_000_000.0


def _priced(model_id: str) -> dict:
    return {"id": model_id, "pricing": {"prompt": "0.000001"}}


def _write(path: Path, at: float | None, models: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {"models": models or []}
    if at is not None:
        doc["at"] = at
    path.write_text(json.dumps(doc), encoding="utf-8")


def _config(tmp_path: Path) -> SiteConfig:
    return SiteConfig(base_dir=str(tmp_path))


def _offline(url: str, timeout: float) -> list[dict]:
    raise TimeoutError("provider unreachable")


def _truncated(url: str, timeout: float) -> list[dict]:
    raise http.client.IncompleteRead(b'{"data": [', 41)


class _StubResponse:
    """The HTTP response ``fetch_models`` reads, carrying *body* verbatim."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _StubResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_table_path_resolves_through_site_config(tmp_path: Path, monkeypatch) -> None:
    """The table path is derived from the site, not a literal in the reader."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    assert provider_prices.table_path(config) == (
        tmp_path / "agents" / "openrouter-prices.json"
    )
    outcome = provider_prices.refresh(
        config=config, fetch=lambda url, timeout: [_priced("m")], now=NOW
    )
    assert outcome.path == tmp_path / "agents" / "openrouter-prices.json"
    assert outcome.path.exists()


def test_watch_module_holds_no_price_path_literal() -> None:
    """``watch.py`` reaches the table through the owning module, not a literal."""
    source = Path(watch.__file__).read_text(encoding="utf-8")
    assert "DEFAULT_PRICE_PATH" not in source
    assert "gpu-watch/openrouter-prices" not in source


def test_young_table_is_not_refetched(tmp_path: Path, monkeypatch) -> None:
    """A table inside the refresh window is served as-is."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    _write(path, NOW - DAY / 2, [_priced("kept")])
    calls: list[tuple[str, float]] = []

    def fetch(url: object, timeout: float) -> list[dict]:
        calls.append((url, timeout))
        return [_priced("unused")]

    outcome = provider_prices.refresh(config=config, fetch=fetch, now=NOW)
    assert outcome.refreshed is False
    assert calls == []
    assert outcome.age == pytest.approx(DAY / 2, abs=1.0)
    doc = json.loads(path.read_text())
    assert [m["id"] for m in doc["models"]] == ["kept"]


def test_old_table_is_refetched_under_a_bounded_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """A table past the window is refetched, and the fetch is bounded."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    _write(path, NOW - DAY * 2, [_priced("old")])
    seen: dict = {}

    def fetch(url: str, timeout: float) -> list[dict]:
        seen["url"], seen["timeout"] = url, timeout
        return [_priced("fresh")]

    outcome = provider_prices.refresh(config=config, fetch=fetch, now=NOW)
    assert outcome.refreshed is True
    assert outcome.age == 0.0
    assert provider_prices.FETCH_TIMEOUT == 10.0
    assert seen["timeout"] == provider_prices.FETCH_TIMEOUT
    assert seen["url"] == provider_prices.PRICE_URL
    doc = json.loads(path.read_text())
    assert [m["id"] for m in doc["models"]] == ["fresh"]
    assert doc["at"] == NOW


def test_a_failed_fetch_keeps_the_cached_table(tmp_path: Path, monkeypatch) -> None:
    """A fetch that fails leaves the cached table in use at its true age."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    _write(path, NOW - DAY * 3, [_priced("cached")])
    before = path.read_bytes()

    outcome = provider_prices.refresh(config=config, fetch=_offline, now=NOW)
    assert outcome.refreshed is False
    assert outcome.error is not None
    assert outcome.age == pytest.approx(DAY * 3, abs=1.0)
    assert path.read_bytes() == before
    assert json.loads(path.read_text())["models"][0]["id"] == "cached"


def test_an_empty_fetch_does_not_overwrite_the_cached_table(
    tmp_path: Path, monkeypatch
) -> None:
    """An empty model list is a failed fetch, not a table of no prices."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    _write(path, NOW - DAY * 3, [_priced("cached")])
    before = path.read_bytes()

    outcome = provider_prices.refresh(
        config=config, fetch=lambda url, timeout: [], now=NOW
    )
    assert outcome.refreshed is False
    assert outcome.error is not None
    assert path.read_bytes() == before


def test_a_truncated_body_keeps_the_cached_table(
    tmp_path: Path, monkeypatch
) -> None:
    """A response cut off mid-body is a failed fetch, not a fetch that escapes."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    _write(path, NOW - DAY * 3, [_priced("cached")])
    before = path.read_bytes()

    outcome = provider_prices.refresh(config=config, fetch=_truncated, now=NOW)
    assert outcome.refreshed is False
    assert outcome.error is not None
    assert outcome.age == pytest.approx(DAY * 3, abs=1.0)
    assert path.read_bytes() == before


def test_a_json_array_body_keeps_the_cached_table(
    tmp_path: Path, monkeypatch
) -> None:
    """A valid-JSON body that is a list, not an object, is a failed fetch."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    _write(path, NOW - DAY * 3, [_priced("cached")])
    before = path.read_bytes()

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: _StubResponse(b"[]")
    )
    outcome = provider_prices.refresh(config=config, now=NOW)
    assert outcome.refreshed is False
    assert outcome.error is not None
    assert outcome.age == pytest.approx(DAY * 3, abs=1.0)
    assert path.read_bytes() == before


def test_write_is_atomic_through_a_scratch_file(tmp_path: Path, monkeypatch) -> None:
    """The write renames a scratch sibling onto the path, never in place."""
    target = tmp_path / "table.json"
    renames: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src: object, dst: object) -> None:
        renames.append((Path(str(src)), Path(str(dst))))
        real_replace(src, dst)

    monkeypatch.setattr(provider_prices.os, "replace", spy)
    provider_prices.write_atomic(target, b'{"at": 1, "models": []}')
    assert renames == [(target.with_suffix(".json.tmp"), target)]
    assert not target.with_suffix(".json.tmp").exists()
    assert target.read_bytes() == b'{"at": 1, "models": []}'


def test_age_comes_from_the_at_field_not_the_mtime(tmp_path: Path) -> None:
    """A copy, restore or touch moves the mtime without moving the fetch."""
    path = tmp_path / "table.json"
    _write(path, NOW - DAY, [])
    os.utime(path, (NOW, NOW))
    doc = json.loads(path.read_text())
    assert provider_prices.table_age(doc, path, now=NOW) == pytest.approx(DAY, abs=1.0)


def test_age_falls_back_to_the_mtime_when_at_is_absent(tmp_path: Path) -> None:
    """Only a table with no recorded fetch falls back to the file's mtime."""
    path = tmp_path / "table.json"
    _write(path, None, [])
    os.utime(path, (NOW, NOW - DAY * 2))
    doc = json.loads(path.read_text())
    assert provider_prices.table_age(doc, path, now=NOW) == pytest.approx(
        DAY * 2, abs=1.0
    )


def test_legacy_table_seeds_the_new_path_once(tmp_path: Path, monkeypatch) -> None:
    """The retired cache is copied while the new path is absent, and once only."""
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    legacy = tmp_path / "legacy.json"
    _write(legacy, NOW - DAY * 10, [_priced("historic")])
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", legacy)

    outcome = provider_prices.refresh(config=config, fetch=_offline, now=NOW)
    assert path.exists()
    assert outcome.age == pytest.approx(DAY * 10, abs=1.0)
    assert json.loads(path.read_text())["models"][0]["id"] == "historic"

    _write(legacy, NOW, [_priced("changed")])
    provider_prices.refresh(config=config, fetch=_offline, now=NOW)
    assert json.loads(path.read_text())["models"][0]["id"] == "historic"


def test_watch_reads_the_table_the_module_owns(tmp_path: Path, monkeypatch) -> None:
    """``watch`` resolves the path through the module's SiteConfig property."""
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(tmp_path))
    path = tmp_path / "agents" / "openrouter-prices.json"
    _write(path, NOW, [_priced("served")])
    assert watch.load_prices() == [_priced("served")]
