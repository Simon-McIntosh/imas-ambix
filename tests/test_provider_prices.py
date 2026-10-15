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
HOUR = 3600.0
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


def test_watch_defines_no_rate_normaliser_of_its_own() -> None:
    """One normaliser, owned here: the ledger imports it rather than repeating it.

    Two normalisers are two answers to what a pricing block charges, and the
    one the ledger prices from would drift from the one the history records.
    """
    source = Path(watch.__file__).read_text(encoding="utf-8")
    assert "def _rates" not in source
    assert "def rates" not in source
    assert "provider_prices.rates" in source
    assert provider_prices.rates({"prompt": 1.0, "completion": 2.0}) == {
        "prompt": 1.0,
        "cache_read": 1.0,
        "completion": 2.0,
    }
    assert provider_prices.rates(
        {"prompt": 1.0, "completion": 2.0}, fallback=3.0
    ) == {"prompt": 1.0, "cache_read": 3.0, "completion": 2.0}


def test_the_recovered_row_cites_the_read_it_was_back_solved_from() -> None:
    """The recovered 2026-10-02 rate cites its read, in that read's units."""
    stamps = [entry["at"] for entry in provider_prices.RECOVERED_OBSERVATIONS]
    row = provider_prices.RECOVERED_OBSERVATIONS[stamps.index(1790935380.0)]
    citation = row["citation"]
    assert "22.8 B input" in citation
    assert "0.166 B output tokens" in citation


def test_young_table_is_not_refetched(tmp_path: Path, monkeypatch) -> None:
    """A table inside the refresh window is served as-is."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    path = provider_prices.table_path(config)
    _write(path, NOW - HOUR / 2, [_priced("kept")])
    calls: list[tuple[str, float]] = []

    def fetch(url: object, timeout: float) -> list[dict]:
        calls.append((url, timeout))
        return [_priced("unused")]

    outcome = provider_prices.refresh(config=config, fetch=fetch, now=NOW)
    assert outcome.refreshed is False
    assert calls == []
    assert outcome.age == pytest.approx(HOUR / 2, abs=1.0)
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


def test_a_truncated_body_keeps_the_cached_table(tmp_path: Path, monkeypatch) -> None:
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


def test_a_json_array_body_keeps_the_cached_table(tmp_path: Path, monkeypatch) -> None:
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


def _model(
    model_id: str,
    prompt: str = "0.000001",
    cache: str = "0.0000001",
    completion: str = "0.000002",
) -> dict:
    """A models row whose three rates are all usable."""
    return {
        "id": model_id,
        "pricing": {
            "prompt": prompt,
            "input_cache_read": cache,
            "completion": completion,
        },
    }


def _endpoint(
    provider: str,
    quantization: str,
    prompt: float,
    cache: float,
    completion: float,
) -> dict:
    """One provider row in the shape ``fetch_endpoints`` returns."""
    return {
        "provider": provider,
        "tag": "",
        "quantization": quantization,
        "pricing": {
            "prompt": prompt,
            "input_cache_read": cache,
            "completion": completion,
        },
    }


def _history(config: SiteConfig) -> Path:
    return provider_prices.history_path(config=config)


def _rows(config: SiteConfig, source: str) -> list[dict]:
    return [
        row
        for row in provider_prices.read_history(_history(config))
        if row.get("source") == source
    ]


def _stub(*rows: dict):
    return lambda url, timeout: list(rows)


def test_refresh_age_is_one_hour() -> None:
    """The rates move within hours, so the table is refetched hourly."""
    assert provider_prices.REFRESH_AGE == 3600.0


def test_identical_refreshes_append_one_line_per_model(
    tmp_path: Path, monkeypatch
) -> None:
    """A repeat of the same prices is not a change, and is not recorded twice."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    stub = _stub(_model("alpha"), _model("beta"))

    provider_prices.refresh(config=config, fetch=stub, now=NOW)
    assert [row["id"] for row in _rows(config, "models")] == ["alpha", "beta"]

    provider_prices.refresh(config=config, fetch=stub, now=NOW + 2 * HOUR)
    rows = _rows(config, "models")
    assert [row["id"] for row in rows] == ["alpha", "beta"]
    assert [row["at"] for row in rows] == [NOW, NOW]


def test_a_changed_price_appends_one_line_for_that_model_only(
    tmp_path: Path, monkeypatch
) -> None:
    """Only the model whose rate moved is recorded again."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)

    provider_prices.refresh(
        config=config, fetch=_stub(_model("alpha"), _model("beta")), now=NOW
    )
    provider_prices.refresh(
        config=config,
        fetch=_stub(_model("alpha", prompt="0.000003"), _model("beta")),
        now=NOW + 2 * HOUR,
    )

    appended = [row for row in _rows(config, "models") if row["at"] != NOW]
    assert [row["id"] for row in appended] == ["alpha"]
    assert appended[0]["pricing"]["prompt"] == "0.000003"


def test_a_failed_fetch_appends_nothing(tmp_path: Path, monkeypatch) -> None:
    """A fetch that published nothing records nothing, on any history."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)

    provider_prices.refresh(config=config, fetch=_offline, now=NOW)
    assert not _history(config).exists()

    provider_prices.refresh(config=config, fetch=_stub(_model("alpha")), now=NOW)
    before = _history(config).read_bytes()

    outcome = provider_prices.refresh(config=config, fetch=_offline, now=NOW + 2 * HOUR)
    assert outcome.refreshed is False
    assert outcome.error is not None
    assert _history(config).read_bytes() == before


def test_the_endpoints_listing_is_fetched_for_the_named_slugs_only(
    tmp_path: Path, monkeypatch
) -> None:
    """A refresh that names no slug fetches no listing, as before."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    slug = "deepseek/deepseek-v4.1-flash"
    listing = [_endpoint("one", "fp8", 1e-07, 1e-08, 1e-06)]
    seen: list[str] = []

    def endpoints_fetch(url: str, timeout: float) -> list[dict]:
        seen.append(url)
        return list(listing)

    provider_prices.refresh(config=config, fetch=_stub(_model("alpha")), now=NOW)
    assert seen == []

    provider_prices.refresh(
        config=config,
        fetch=_stub(_model("alpha")),
        endpoints_fetch=endpoints_fetch,
        slugs=[slug],
        now=NOW + 2 * HOUR,
    )
    assert seen == [provider_prices.endpoints_url(slug)]
    assert [row["id"] for row in _rows(config, "endpoints")] == [slug]

    provider_prices.refresh(
        config=config,
        fetch=_stub(_model("alpha")),
        endpoints_fetch=endpoints_fetch,
        slugs=[slug],
        now=NOW + 4 * HOUR,
    )
    assert len(seen) == 2
    assert len(_rows(config, "endpoints")) == 1

    listing.append(_endpoint("two", "fp4", 2e-07, 2e-07, 2e-07))
    provider_prices.refresh(
        config=config,
        fetch=_stub(_model("alpha")),
        endpoints_fetch=endpoints_fetch,
        slugs=[slug],
        now=NOW + 6 * HOUR,
    )
    assert len(_rows(config, "endpoints")) == 2


def test_an_endpoints_row_is_recorded_only_when_a_price_moves(
    tmp_path: Path, monkeypatch
) -> None:
    """A listing that reordered, or renamed a provider, has not repriced."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    slug = "deepseek/deepseek-v4.1-flash"
    listing = [
        _endpoint("one", "fp8", 1e-07, 1e-08, 1e-06),
        _endpoint("two", "fp8", 2e-07, 2e-08, 2e-06),
    ]
    served: list[list[dict]] = [list(listing)]

    def endpoints_fetch(url: str, timeout: float) -> list[dict]:
        return [dict(row) for row in served[0]]

    def refresh(at: float) -> int:
        provider_prices.refresh(
            config=config,
            fetch=_stub(_model("alpha")),
            endpoints_fetch=endpoints_fetch,
            slugs=[slug],
            now=at,
        )
        return len(_rows(config, "endpoints"))

    assert refresh(NOW) == 1

    served[0] = list(reversed(listing))
    assert refresh(NOW + 2 * HOUR) == 1

    served[0] = [
        {**row, "tag": "renamed"} for row in reversed(listing)
    ]
    assert refresh(NOW + 4 * HOUR) == 1

    served[0] = [
        {**row, "pricing": {**row["pricing"], "prompt": 5e-07}}
        for row in listing
    ]
    assert refresh(NOW + 6 * HOUR) == 2


def test_rate_in_force_returns_the_last_headline_observation() -> None:
    """The readers pick the last row at or before the instant, by source."""
    history = [
        {
            "at": 100.0,
            "id": "s",
            "source": "recovered",
            "pricing": {
                "prompt": 3e-07,
                "input_cache_read": 6e-09,
                "completion": 1.2e-06,
            },
        },
        {
            "at": 300.0,
            "id": "s",
            "source": "models",
            "pricing": {
                "prompt": 2e-07,
                "input_cache_read": 2e-08,
                "completion": 1e-06,
            },
        },
        {
            "at": 200.0,
            "id": "s",
            "source": "endpoints",
            "endpoints": [_endpoint("one", "fp8", 5e-07, 5e-08, 5e-06)],
        },
    ]
    assert provider_prices.rate_in_force(history, "s", 99.0) is None
    assert provider_prices.rate_in_force(history, "s", 100.0)["prompt"] == 3e-07
    at_two_hundred = provider_prices.rate_in_force(history, "s", 200.0)
    assert at_two_hundred["prompt"] == 3e-07
    at_four_hundred = provider_prices.rate_in_force(history, "s", 400.0)
    assert at_four_hundred == {
        "prompt": 2e-07,
        "cache_read": 2e-08,
        "completion": 1e-06,
    }
    assert provider_prices.rate_in_force(history, "other", 400.0) is None


def test_rate_in_force_reads_a_history_path(tmp_path: Path) -> None:
    """A path and an already-read history answer the same question."""
    path = tmp_path / "history.jsonl"
    row = {
        "at": 10.0,
        "id": "s",
        "source": "models",
        "pricing": {"prompt": 4e-08, "input_cache_read": 4e-09, "completion": 4e-07},
    }
    provider_prices.append_history(path, [row])
    assert provider_prices.rate_in_force(path, "s", 20.0) == {
        "prompt": 4e-08,
        "cache_read": 4e-09,
        "completion": 4e-07,
    }


def test_provider_median_in_force_excludes_narrow_quantisations() -> None:
    """The median answers for the eight-bit-or-better routes only."""
    history = [
        {
            "at": 100.0,
            "id": "s",
            "source": "endpoints",
            "endpoints": [
                _endpoint("a", "fp8", 1e-07, 1e-08, 1e-06),
                _endpoint("b", "bf16", 2e-07, 2e-08, 2e-06),
                _endpoint("c", "fp16", 3e-07, 3e-08, 3e-06),
                _endpoint("d", "fp4", 1e-09, 1e-10, 1e-08),
                _endpoint("e", "int4", 2e-09, 2e-10, 2e-08),
            ],
        }
    ]
    assert provider_prices.provider_median_in_force(history, "s", 99.0) is None
    median = provider_prices.provider_median_in_force(history, "s", 100.0)
    assert median["prompt"] == pytest.approx(2e-07)
    assert median["cache_read"] == pytest.approx(2e-08)
    assert median["completion"] == pytest.approx(2e-06)


def test_provider_median_in_force_is_none_when_nothing_survives() -> None:
    """A listing of narrow routes only has no median to report."""
    history = [
        {
            "at": 1.0,
            "id": "s",
            "source": "endpoints",
            "endpoints": [
                _endpoint("d", "fp4", 1e-09, 1e-10, 1e-08),
                _endpoint("e", "int4", 2e-09, 2e-10, 2e-08),
            ],
        }
    ]
    assert provider_prices.provider_median_in_force(history, "s", 10.0) is None


def test_an_endpoint_declaring_no_quantisation_is_kept() -> None:
    """Silence about precision is not a claim of a narrow one."""
    history = [
        {
            "at": 1.0,
            "id": "s",
            "source": "endpoints",
            "endpoints": [_endpoint("a", "", 7e-08, 7e-09, 7e-07)],
        }
    ]
    median = provider_prices.provider_median_in_force(history, "s", 10.0)
    assert median == {
        "prompt": pytest.approx(7e-08),
        "cache_read": pytest.approx(7e-09),
        "completion": pytest.approx(7e-07),
    }


def test_a_first_refresh_records_the_recovered_observations_once(
    tmp_path: Path, monkeypatch
) -> None:
    """The rates known before the history existed are recorded, then never again."""
    monkeypatch.setattr(provider_prices, "LEGACY_PRICE_PATH", tmp_path / "absent.json")
    config = _config(tmp_path)
    stub = _stub(_model("alpha"), _model("beta"))

    provider_prices.refresh(config=config, fetch=stub, now=NOW)
    rows = provider_prices.read_history(_history(config))
    recovered = [
        row for row in rows if row["source"] == "recovered" and row["citation"]
    ]
    assert len(recovered) == 6
    assert {row["id"] for row in recovered} == {provider_prices.RECOVERED_SLUG}
    stamps = [row["at"] for row in recovered]
    assert stamps == sorted(stamps)
    assert stamps[-1] < NOW
    assert recovered[-1]["pricing"]["prompt"] == pytest.approx(2.016e-08)

    first_fetched = next(
        index for index, row in enumerate(rows) if row["source"] == "models"
    )
    assert all(
        index < first_fetched
        for index, row in enumerate(rows)
        if row["source"] == "recovered"
    )

    provider_prices.refresh(config=config, fetch=stub, now=NOW + 2 * HOUR)
    assert len(_rows(config, "recovered")) == 6
