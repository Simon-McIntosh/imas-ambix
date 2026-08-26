"""Tests for benchmark provenance capture and speculative-decode acceptance.

Coverage
--------
1.  Provenance is fully populated from a model profile, every key present.
2.  Provenance degrades to ``None`` values — not a crash, not a missing key —
    when no profile is supplied.
3.  The Prometheus parser pulls speculative counters out of a realistic text
    exposition scrape and skips unrelated metrics and comments.
4.  Acceptance rate is the accepted/draft ratio over the run window.
5.  A zero draft-token window yields no rate instead of dividing by zero.
6.  A 404 on ``/metrics`` reports ``unavailable`` without raising.
7.  A connection error on ``/metrics`` reports ``unavailable``.
8.  A server-reported context window wins over the profile's request, which is
    preserved beside it; agreeing values add no second key.
9.  ``run_benchmark`` keeps the raw ``/v1/models`` keys and adds the three
    provenance keys beside them.
10. The throughput category brackets the scrape so acceptance is measured
    across it.

All HTTP is stubbed at ``urllib.request.urlopen`` — no network, no GPU.
"""

from __future__ import annotations

import json
import urllib.error
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from imas_ambix.agent import bench
from imas_ambix.agent.profile import (
    EngineConfig,
    ModelConfig,
    ModelProfile,
    SlurmDefaults,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Every key a rendered report may address. A missing key is as much a defect
# as a wrong value, because a renderer cannot distinguish it from a bad read.
PROVENANCE_KEYS = frozenset(
    {
        "profile_slug",
        "model_name",
        "served_name",
        "engine_type",
        "engine_version",
        "tensor_parallel",
        "gpus",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "speculative_method",
        "speculative_num_tokens",
        "quantization",
        "slurm_job_id",
        "gpu_host",
        "captured_at",
    }
)

SPEC_DECODE_KEYS = frozenset(
    {
        "draft_tokens_total",
        "accepted_tokens_total",
        "acceptance_rate",
        "num_accepted_per_pos",
        "source",
    }
)

SERVED_URL = "http://gpu-node.example:18800"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _stub_urlopen(
    routes: dict[str, list[bytes | Exception]],
) -> Callable[..., _FakeResponse]:
    """Return a ``urlopen`` replacement dispatching on URL suffix.

    Each route holds the outcomes for successive calls; a single-entry route
    answers every call the same way, so a multi-entry route is how a test
    makes a counter advance between two scrapes. An unrouted URL answers 404,
    which is how an engine without the route behaves.
    """
    pending = {suffix: list(outcomes) for suffix, outcomes in routes.items()}

    def _urlopen(req: Any, timeout: float | None = None, **_kwargs: Any) -> Any:
        url = getattr(req, "full_url", None) or str(req)
        for suffix, outcomes in pending.items():
            if url.endswith(suffix):
                outcome = outcomes[0] if len(outcomes) == 1 else outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return _FakeResponse(outcome)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    return _urlopen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profile() -> ModelProfile:
    """A profile exercising every provenance field the profile can supply."""
    return ModelProfile(
        slug="test-model",
        model=ModelConfig(
            name="Test Model",
            hf_repo="test-org/test-model",
            served_name="test-org/test-model",
            size_gb=744,
            max_context=229376,
        ),
        engine=EngineConfig(
            type="vllm",
            tensor_parallel=8,
            kv_cache_dtype="fp8_e4m3",
            max_total_tokens=229376,
            max_num_seqs=1024,
            speculative_method="mtp",
            speculative_num_tokens=5,
        ),
        slurm=SlurmDefaults(gpus=8),
    )


@pytest.fixture(autouse=True)
def _clear_slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make scheduler-derived fields deterministic regardless of the shell."""
    for var in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURMD_NODENAME"):
        monkeypatch.delenv(var, raising=False)


def _models_payload(**extra: Any) -> bytes:
    """A ``/v1/models`` payload shaped like the one vLLM returns."""
    card: dict[str, Any] = {
        "id": "test-org/test-model",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "vllm",
        "root": "/work/model",
        "parent": None,
        "permission": [],
    }
    card.update(extra)
    return json.dumps({"object": "list", "data": [card]}).encode()


def _metrics_text(
    *,
    drafts: float,
    draft_tokens: float,
    accepted_tokens: float,
    per_pos: tuple[float, ...] = (),
) -> bytes:
    """A Prometheus scrape shaped like a vLLM speculative-decode endpoint."""
    lines = [
        "# HELP vllm:num_requests_running Number of running requests.",
        "# TYPE vllm:num_requests_running gauge",
        'vllm:num_requests_running{engine="0",model_name="test"} 3.0',
        "# HELP vllm:spec_decode_num_drafts_total Number of drafts.",
        "# TYPE vllm:spec_decode_num_drafts_total counter",
        f'vllm:spec_decode_num_drafts_total{{engine="0"}} {drafts}',
        "# HELP vllm:spec_decode_num_draft_tokens_total Draft tokens proposed.",
        "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
        f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {draft_tokens}',
        "# HELP vllm:spec_decode_num_accepted_tokens_total Tokens kept.",
        "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted_tokens}',
        "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter",
    ]
    lines += [
        f'vllm:spec_decode_num_accepted_tokens_per_pos{{engine="0",position="{i}"}} {v}'
        for i, v in enumerate(per_pos)
    ]
    lines.append("")
    return "\n".join(lines).encode()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_populated_from_profile(profile: ModelProfile) -> None:
    """Every provenance key is present and profile fields land in it."""
    routes = {
        "/version": [json.dumps({"version": "0.23.0"}).encode()],
        "/v1/models": [_models_payload()],
    }
    with patch("urllib.request.urlopen", _stub_urlopen(routes)):
        provenance = bench.capture_provenance(
            SERVED_URL,
            profile=profile,
            models_payload=json.loads(_models_payload()),
        )

    assert set(provenance) >= PROVENANCE_KEYS
    assert provenance["profile_slug"] == "test-model"
    assert provenance["model_name"] == "Test Model"
    assert provenance["served_name"] == "test-org/test-model"
    assert provenance["engine_type"] == "vllm"
    assert provenance["engine_version"] == "0.23.0"
    assert provenance["tensor_parallel"] == 8
    assert provenance["gpus"] == 8
    assert provenance["kv_cache_dtype"] == "fp8_e4m3"
    assert provenance["max_model_len"] == 229376
    assert provenance["max_num_seqs"] == 1024
    assert provenance["speculative_method"] == "mtp"
    assert provenance["speculative_num_tokens"] == 5
    assert provenance["gpu_host"] == "gpu-node.example"
    assert provenance["captured_at"].endswith("+00:00")


def test_provenance_reads_slurm_job_from_environment(
    profile: ModelProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-allocation run records the job id that produced the numbers."""
    monkeypatch.setenv("SLURM_JOB_ID", "1222821")
    with patch("urllib.request.urlopen", _stub_urlopen({})):
        provenance = bench.capture_provenance(SERVED_URL, profile=profile)
    assert provenance["slurm_job_id"] == "1222821"


def test_provenance_degrades_to_none_without_profile() -> None:
    """Without a profile the keys remain, holding ``None`` rather than guesses."""
    with patch("urllib.request.urlopen", _stub_urlopen({})):
        provenance = bench.capture_provenance("http://localhost:18800")

    assert set(provenance) >= PROVENANCE_KEYS
    unresolved = PROVENANCE_KEYS - {"captured_at"}
    assert {key: provenance[key] for key in unresolved} == dict.fromkeys(unresolved)
    assert provenance["captured_at"]


def test_server_context_window_wins_and_keeps_the_request(
    profile: ModelProfile,
) -> None:
    """A disagreeing server value wins and the request survives beside it."""
    payload = json.loads(_models_payload(max_model_len=262144))
    with patch("urllib.request.urlopen", _stub_urlopen({})):
        provenance = bench.capture_provenance(
            SERVED_URL, profile=profile, models_payload=payload
        )

    assert provenance["max_model_len"] == 262144
    assert provenance["max_model_len_profile"] == 229376


def test_agreeing_values_add_no_second_key(profile: ModelProfile) -> None:
    """Server and profile agreeing leaves a single, unambiguous value."""
    payload = json.loads(_models_payload(max_model_len=229376))
    with patch("urllib.request.urlopen", _stub_urlopen({})):
        provenance = bench.capture_provenance(
            SERVED_URL, profile=profile, models_payload=payload
        )

    assert provenance["max_model_len"] == 229376
    assert "max_model_len_profile" not in provenance


def test_server_reported_quantization_is_kept(profile: ModelProfile) -> None:
    """A quantization the engine reports is recorded; the profile has none."""
    payload = json.loads(_models_payload(quantization="fp8"))
    with patch("urllib.request.urlopen", _stub_urlopen({})):
        provenance = bench.capture_provenance(
            SERVED_URL, profile=profile, models_payload=payload
        )
    assert provenance["quantization"] == "fp8"


# ---------------------------------------------------------------------------
# Prometheus parsing and acceptance
# ---------------------------------------------------------------------------


def test_snapshot_extracts_counters_from_a_real_scrape() -> None:
    """Token counters are read out; unrelated metrics and comments are skipped."""
    text = _metrics_text(
        drafts=100.0,
        draft_tokens=500.0,
        accepted_tokens=400.0,
        per_pos=(180.0, 130.0, 90.0),
    ).decode()
    snapshot = bench._spec_decode_snapshot(text)

    # 500, not the 100 drafts: the token counter is preferred over a
    # speculative counter that does not count tokens.
    assert snapshot["draft_tokens_total"] == 500.0
    assert snapshot["accepted_tokens_total"] == 400.0
    assert snapshot["num_accepted_per_pos"] == [180.0, 130.0, 90.0]


def test_parser_sums_across_label_sets() -> None:
    """One counter exposed per engine totals across the label sets."""
    text = "\n".join(
        [
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 300.0',
            'vllm:spec_decode_num_draft_tokens_total{engine="1"} 200.0',
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 240.0',
            'vllm:spec_decode_num_accepted_tokens_total{engine="1"} 160.0',
        ]
    )
    snapshot = bench._spec_decode_snapshot(text)
    assert snapshot["draft_tokens_total"] == 500.0
    assert snapshot["accepted_tokens_total"] == 400.0


def test_parser_tolerates_a_malformed_line() -> None:
    """A bad value discards its own line, not the counters around it."""
    text = "\n".join(
        [
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} not-a-number',
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 400.0',
        ]
    )
    snapshot = bench._spec_decode_snapshot(text)
    assert snapshot["draft_tokens_total"] is None
    assert snapshot["accepted_tokens_total"] == 400.0


def test_acceptance_rate_is_the_ratio_over_the_window() -> None:
    """Cumulative counters are differenced, so the rate covers this run only."""
    before = bench._spec_decode_snapshot(
        _metrics_text(
            drafts=100.0,
            draft_tokens=500.0,
            accepted_tokens=400.0,
            per_pos=(180.0, 130.0, 90.0),
        ).decode()
    )
    after = bench._spec_decode_snapshot(
        _metrics_text(
            drafts=300.0,
            draft_tokens=1500.0,
            accepted_tokens=1200.0,
            per_pos=(520.0, 400.0, 280.0),
        ).decode()
    )
    window = bench._spec_decode_window(before, after)

    assert set(window) >= SPEC_DECODE_KEYS
    assert window["source"] == "metrics"
    assert window["draft_tokens_total"] == 1000
    assert window["accepted_tokens_total"] == 800
    assert window["acceptance_rate"] == pytest.approx(0.8)
    assert window["num_accepted_per_pos"] == [340, 270, 190]


def test_zero_draft_tokens_yields_no_rate() -> None:
    """An idle window reports no rate instead of dividing by zero."""
    text = _metrics_text(
        drafts=0.0, draft_tokens=0.0, accepted_tokens=0.0, per_pos=(0.0,)
    ).decode()
    snapshot = bench._spec_decode_snapshot(text)
    window = bench._spec_decode_window(snapshot, dict(snapshot))

    assert window["draft_tokens_total"] == 0
    assert window["acceptance_rate"] is None
    assert window["source"] == "metrics"


def test_counter_reset_discards_the_window() -> None:
    """A restarted engine resets its counters, so the window means nothing."""
    before = {"draft_tokens_total": 5000.0, "accepted_tokens_total": 4000.0}
    after = {"draft_tokens_total": 10.0, "accepted_tokens_total": 8.0}
    window = bench._spec_decode_window(before, after)
    assert window["source"] == "unavailable"
    assert window["acceptance_rate"] is None


def test_scrape_of_missing_endpoint_reports_unavailable() -> None:
    """A 404 on /metrics is reported, not raised."""
    with patch("urllib.request.urlopen", _stub_urlopen({})):
        snapshot = bench._scrape_spec_decode(SERVED_URL)
    assert snapshot is None

    window = bench._spec_decode_window(snapshot, snapshot)
    assert window == {
        "draft_tokens_total": None,
        "accepted_tokens_total": None,
        "acceptance_rate": None,
        "num_accepted_per_pos": None,
        "source": "unavailable",
    }


def test_scrape_of_unreachable_server_reports_unavailable() -> None:
    """A connection error is reported, not raised."""
    routes: dict[str, list[bytes | Exception]] = {
        "/metrics": [urllib.error.URLError("connection refused")]
    }
    with patch("urllib.request.urlopen", _stub_urlopen(routes)):
        snapshot = bench._scrape_spec_decode(SERVED_URL)
    assert snapshot is None
    assert bench._spec_decode_window(snapshot, snapshot)["source"] == "unavailable"


def test_engine_without_speculative_counters_reports_unavailable() -> None:
    """A served endpoint with no speculative metrics carries no acceptance."""
    routes: dict[str, list[bytes | Exception]] = {
        "/metrics": [b'vllm:num_requests_running{engine="0"} 1.0\n']
    }
    with patch("urllib.request.urlopen", _stub_urlopen(routes)):
        snapshot = bench._scrape_spec_decode(SERVED_URL)
    assert snapshot is not None
    assert bench._spec_decode_window(snapshot, snapshot)["source"] == "unavailable"


# ---------------------------------------------------------------------------
# Report wiring
# ---------------------------------------------------------------------------


def test_report_keeps_raw_model_keys_and_adds_provenance(
    profile: ModelProfile,
) -> None:
    """The added keys sit beside the raw payload, so old readers still read."""
    routes: dict[str, list[bytes | Exception]] = {
        "/v1/models": [_models_payload()],
        "/version": [json.dumps({"version": "0.23.0"}).encode()],
    }
    with patch("urllib.request.urlopen", _stub_urlopen(routes)):
        report = bench.run_benchmark(
            SERVED_URL,
            "test-org/test-model",
            categories=["unrouted-category"],
            profile=profile,
        )

    info = report.server_info
    assert info["object"] == "list"
    assert info["data"][0]["id"] == "test-org/test-model"
    assert info["models"]["data"] == info["data"]
    assert info["provenance"]["profile_slug"] == "test-model"
    assert info["spec_decode"]["source"] == "unavailable"
    # The report must still serialize with the added keys in place.
    assert json.loads(report.to_json())["server_info"]["provenance"]["gpus"] == 8


def test_throughput_category_brackets_the_acceptance_scrape(
    profile: ModelProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance is measured across the decode-heavy category, not lifetime."""
    monkeypatch.setattr(
        bench, "_run_throughput", lambda *_args, **_kwargs: [bench.BenchResult()]
    )
    routes: dict[str, list[bytes | Exception]] = {
        "/v1/models": [_models_payload()],
        "/metrics": [
            _metrics_text(drafts=1.0, draft_tokens=500.0, accepted_tokens=400.0),
            _metrics_text(drafts=3.0, draft_tokens=1500.0, accepted_tokens=1200.0),
        ],
    }
    with patch("urllib.request.urlopen", _stub_urlopen(routes)):
        report = bench.run_benchmark(
            SERVED_URL,
            "test-org/test-model",
            categories=["throughput"],
            profile=profile,
        )

    spec = report.server_info["spec_decode"]
    assert spec["source"] == "metrics"
    assert spec["draft_tokens_total"] == 1000
    assert spec["accepted_tokens_total"] == 800
    assert spec["acceptance_rate"] == pytest.approx(0.8)
