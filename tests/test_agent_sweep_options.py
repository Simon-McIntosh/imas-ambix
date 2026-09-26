"""Sweep tooling: a caller-supplied concurrency ladder and a DSpark depth override.

A speculative-depth sweep has to vary two things the CLI did not expose. The
concurrency ladder was a module constant, so a sweep could not resolve the
plank around a measured knee; and the DSpark block size lived only in the
profile, so each depth cell cost a profile edit. Both now come from the
incommand line.

Two properties are checked together throughout, because only the pair is
useful: a supplied value reaches the code that acts on it, and an omitted one
leaves the default byte-for-byte as it was. A sweep tool that changes the
default script would move the thing it is trying to measure.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from imas_ambix.cli import main

#: An SGLang profile that declares DSpark, so the block-size override applies.
_DSPARK_PROFILE = "deepseek-v4-1-flash"
#: A vLLM profile, so the override must refuse rather than emit a dead flag.
_VLLM_PROFILE = "deepseek-v4-flash"


class _SchedulerRecorder:
    """Stand in for ``subprocess``, answering ``squeue`` and refusing anything else.

    A dry run is allowed to ask the scheduler what is running; a cancelling or
    submitting verb aborts the command instead of being mistaken for a pass.
    """

    def __init__(self, squeue_stdout: str = "") -> None:
        self.argv: list[list[str]] = []
        self._squeue_stdout = squeue_stdout

    def run(self, argv, **kwargs):  # noqa: ANN001, ANN003 - subprocess.run shape
        argv = list(argv)
        self.argv.append(argv)
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, self._squeue_stdout, "")
        raise AssertionError(f"a dry run issued an unexpected command: {argv}")

    @property
    def destructive(self) -> list[list[str]]:
        """Any verb that would have cancelled or submitted work; empty on a dry run."""
        return [
            argv
            for argv in self.argv
            if argv[0] in {"scancel", "sbatch", "srun", "scontrol", "systemctl"}
        ]


@pytest.fixture
def dry_site(tmp_path, monkeypatch):
    """A site the dry runs can render against without touching a real lane."""
    endpoint = tmp_path / "public" / "endpoints.json"
    endpoint.parent.mkdir(parents=True)
    endpoint.write_text('{"endpoints": []}\n', encoding="utf-8")

    base = tmp_path / "base"
    (base / "agents" / "serve-registry").mkdir(parents=True)

    monkeypatch.setenv("AMBIX_AGENT_ENDPOINT_DOCUMENT", str(endpoint))
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(base))
    monkeypatch.setenv("USER", "operator")
    monkeypatch.delenv("AMBIX_AGENT_API_KEY", raising=False)

    recorder = _SchedulerRecorder()
    monkeypatch.setattr("imas_ambix.agent.cli.subprocess", recorder)
    monkeypatch.setattr("imas_ambix.agent.slurm.subprocess", recorder)
    return SimpleNamespace(recorder=recorder, endpoint=endpoint, base=base)


def _dry_run_script(result) -> str:
    """The generated script from a dry-run output, from its batch header on."""
    marker = result.output.index("#SBATCH")
    return result.output[marker:]


# ── The concurrency ladder ──────────────────────────────────────────


def test_levels_option_reaches_the_bench_runner(monkeypatch) -> None:
    """A ``--levels`` ladder arrives at the benchmark as a parsed ladder.

    The CLI parses the string and hands the runner a list, so a later reader of
    the run's provenance sees the ladder that actually ran rather than the one
    the command line spelled.
    """
    captured: dict[str, object] = {}

    from imas_ambix.agent.bench import BenchReport

    def _run_benchmark(*args, **kwargs):  # noqa: ANN001, ANN003 - recorder
        captured.update(kwargs)
        captured["_args"] = args
        return BenchReport()

    monkeypatch.setattr("imas_ambix.agent.bench.run_benchmark", _run_benchmark)
    monkeypatch.setattr("urllib.request.urlopen", _open_ok)

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "bench",
            "--url",
            "http://stub.invalid:18800",
            "--model",
            "stub-model",
            "--levels",
            "4,8,12,16,20,24,28,32,36",
            "--json",
            "--no-save",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["concurrency_levels"] == [4, 8, 12, 16, 20, 24, 28, 32, 36]


def test_omitting_levels_leaves_the_runner_its_own_ladder(monkeypatch) -> None:
    """No ``--levels`` passes ``None`` through, so the default ladder stands."""
    captured: dict[str, object] = {}

    from imas_ambix.agent.bench import BenchReport

    def _run_benchmark(*args, **kwargs):  # noqa: ANN001, ANN003
        captured.update(kwargs)
        return BenchReport()

    monkeypatch.setattr("imas_ambix.agent.bench.run_benchmark", _run_benchmark)
    monkeypatch.setattr("urllib.request.urlopen", _open_ok)

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "bench",
            "--url",
            "http://stub.invalid:18800",
            "--model",
            "stub-model",
            "--json",
            "--no-save",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["concurrency_levels"] is None


def test_concurrency_runner_sweeps_the_supplied_ladder(monkeypatch) -> None:
    """The runner iterates the ladder it was given, in order.

    Exercised through the runner itself rather than through the failed requests
    it would issue: the ladder is what the sweep varies, and a stub that answers
    every level lets it be read off the rows.
    """
    from imas_ambix.agent import bench as bench_mod

    monkeypatch.setattr(bench_mod, "_stream_chat", _stub_stream_chat)
    monkeypatch.setattr(
        bench_mod, "_scrape_spec_decode", lambda *a, **k: {"accept_length": 3.7}
    )

    results = bench_mod._run_concurrency(
        "http://stub.invalid:18800",
        "stub-model",
        1,
        levels=[4, 8, 12],
    )

    ladder = sorted({r.metadata["n_workers"] for r in results})
    assert ladder == [4, 8, 12]


def test_concurrency_runner_defaults_to_the_hardcoded_ladder(monkeypatch) -> None:
    """With no ladder supplied the runner sweeps the declared default."""
    from imas_ambix.agent import bench as bench_mod

    monkeypatch.setattr(bench_mod, "_stream_chat", _stub_stream_chat)
    monkeypatch.setattr(
        bench_mod, "_scrape_spec_decode", lambda *a, **k: {"accept_length": 3.7}
    )

    results = bench_mod._run_concurrency("http://stub.invalid:18800", "stub-model", 1)

    ladder = sorted({r.metadata["n_workers"] for r in results})
    assert ladder == list(bench_mod.DEFAULT_CONCURRENCY_LEVELS)


def test_each_concurrency_row_records_the_effective_draft_depth(monkeypatch) -> None:
    """Every row carries the engine's effective depth, not the profile's setting.

    A row's throughput is only comparable against the depth the engine was
    actually running, so the value is read from the engine per level and stamped
    on each worker's row.
    """
    from imas_ambix.agent import bench as bench_mod

    monkeypatch.setattr(bench_mod, "_stream_chat", _stub_stream_chat)
    monkeypatch.setattr(
        bench_mod,
        "_scrape_spec_decode",
        lambda *a, **k: {"accept_length": 3.7, "active_draft_tokens": 6.0},
    )

    results = bench_mod._run_concurrency(
        "http://stub.invalid:18800", "stub-model", 1, levels=[4]
    )

    assert results
    for row in results:
        assert row.metadata["spec_decode"] == {
            "accept_length": 3.7,
            "active_draft_tokens": 6.0,
        }


def test_a_row_records_none_when_the_engine_publishes_no_depth(monkeypatch) -> None:
    """An unreachable metrics endpoint is an absence, not a zero."""
    from imas_ambix.agent import bench as bench_mod

    monkeypatch.setattr(bench_mod, "_stream_chat", _stub_stream_chat)
    monkeypatch.setattr(bench_mod, "_scrape_spec_decode", lambda *a, **k: None)

    results = bench_mod._run_concurrency(
        "http://stub.invalid:18800", "stub-model", 1, levels=[4]
    )

    assert results
    for row in results:
        assert row.metadata["spec_decode"] == {
            "accept_length": None,
            "active_draft_tokens": None,
        }


def test_spec_decode_depth_reads_the_sglang_gauges() -> None:
    """The two named quantities come from the SGLang gauges verbatim."""
    from imas_ambix.agent.bench import _spec_decode_depth, _spec_decode_snapshot

    body = "\n".join(
        [
            "# HELP sglang:spec_accept_length Mean accepted length per forward",
            "# TYPE sglang:spec_accept_length gauge",
            "sglang:spec_accept_length 3.68",
            "sglang:spec_num_draft_tokens 6.0",
        ]
    )

    depth = _spec_decode_depth(_spec_decode_snapshot(body))

    assert depth["accept_length"] == pytest.approx(3.68)
    assert depth["active_draft_tokens"] == pytest.approx(6.0)


def test_spec_decode_depth_leaves_unpublished_gauges_unset() -> None:
    """An engine family that publishes neither quantity records ``None``.

    vLLM exposes no acceptance gauge, so the keys exist and are null rather
    than the mapping being absent: a consumer reading a sweep table, where a
    vLLM cell sits beside an SGLang one, sees the same shape in both.
    """
    from imas_ambix.agent.bench import _spec_decode_depth, _spec_decode_snapshot

    body = "\n".join(
        [
            "vllm:num_requests_running 3.0",
            "vllm:gpu_cache_usage_perc 0.4",
        ]
    )

    depth = _spec_decode_depth(_spec_decode_snapshot(body))

    assert depth == {"accept_length": None, "active_draft_tokens": None}


@pytest.mark.parametrize(
    "ladder",
    ["4,eight", "4,0", "4,-2", ",,", "", "4,,8", "4,8,", ",4,8"],
)
def test_malformed_ladder_is_refused(ladder: str) -> None:
    """A malformed ladder is refused whole rather than silently shortened.

    A dropped level would leave a hole that reads as a concurrency the sweep
    decided against rather than as one it never measured.
    """
    from imas_ambix.agent.cli import _concurrency_levels

    with pytest.raises(click.BadParameter):
        _concurrency_levels(ladder)


@pytest.mark.parametrize("ladder", ["4,,8", "4,8,", ",4,8", "4, ,8", ",,"])
def test_blank_ladder_component_is_named_in_the_refusal(ladder: str) -> None:
    """A blank component is refused as an empty entry, not skipped.

    ``4,,8`` must not read as ``[4, 8]``: the interior blank is a level the
    caller did not name, so absorbing it would record that concurrency as one
    the sweep decided against rather than as one it never measured.
    """
    from imas_ambix.agent.cli import _concurrency_levels

    with pytest.raises(click.BadParameter, match="empty component"):
        _concurrency_levels(ladder)


# ── The DSpark block-size override ──────────────────────────────────


def test_restart_dry_run_carries_the_block_size_override(dry_site) -> None:
    """``agent restart`` renders the requested depth into the serve script."""
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "restart",
            _DSPARK_PROFILE,
            "--dry-run",
            "--dspark-block-size",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--speculative-dspark-block-size 4" in result.output
    assert dry_site.recorder.destructive == []


def test_restart_dry_run_default_script_is_unchanged(dry_site) -> None:
    """The default script renders the profile's own declared depth.

    Guarding the default is half of what this node is for: a sweep tool that
    moved the script when unused would change the baseline it exists to measure.
    Slurm argument rendering is untouched by the option, so the profile's
    declared ``3`` is what a plain launch carries.
    """
    default = CliRunner().invoke(
        main, ["agent", "restart", _DSPARK_PROFILE, "--dry-run"]
    )
    overridden = CliRunner().invoke(
        main,
        ["agent", "restart", _DSPARK_PROFILE, "--dry-run", "--dspark-block-size", "5"],
    )

    assert default.exit_code == 0, default.output
    assert overridden.exit_code == 0, overridden.output

    default_script = _dry_run_script(default)
    override_script = _dry_run_script(overridden)

    assert "--speculative-dspark-block-size 3" in default_script
    # The override is exactly one token of the script: everything else, the
    # batch header and every other engine argument, is untouched.
    assert override_script == default_script.replace(
        "--speculative-dspark-block-size 3", "--speculative-dspark-block-size 5"
    )


def test_block_size_override_is_refused_without_dspark(dry_site) -> None:
    """A profile not serving DSpark on SGLang refuses the override.

    The block size is a DSpark setting, so applied to another engine it would
    emit a flag nothing reads and a sweep cell would be labelled with a depth it
    never ran.
    """
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "restart",
            _VLLM_PROFILE,
            "--dry-run",
            "--dspark-block-size",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "needs a profile serving DSpark on SGLang" in result.output


def test_no_speculative_and_block_size_are_mutually_exclusive(dry_site) -> None:
    """The two ask for opposite things, so the pair is refused, not ordered."""
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "restart",
            _DSPARK_PROFILE,
            "--dry-run",
            "--no-speculative",
            "--dspark-block-size",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "cannot be combined with" in result.output


def test_no_speculative_clears_the_sglang_dspark_keys() -> None:
    """``--no-speculative`` disables drafting on either engine family.

    The two families spell speculation differently. Clearing only vLLM's keys
    left an SGLang profile still rendering its DSpark flags, so a serve asked
    for as "off" drafted -- and the sweep's off arm was unreachable.
    """
    from imas_ambix.agent.cli import _without_speculation
    from imas_ambix.agent.profile import load_profile

    profile = load_profile(_DSPARK_PROFILE)
    assert profile.engine.speculative_algorithm == "DSPARK"

    cleared = _without_speculation(profile)

    assert cleared.engine.speculative_algorithm is None
    assert cleared.engine.speculative_dspark_block_size is None
    assert cleared.engine.speculative_method is None
    assert cleared.engine.speculative_num_tokens is None
    assert profile.engine.speculative_algorithm == "DSPARK"  # input untouched


def test_serve_dry_run_offers_the_same_override(dry_site) -> None:
    """``serve`` carries the override too, so both launch paths agree."""
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "serve",
            _DSPARK_PROFILE,
            "--dry-run",
            "--dspark-block-size",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--speculative-dspark-block-size 4" in result.output


# ── Stubs ───────────────────────────────────────────────────────────


def _open_ok(*args, **kwargs):  # noqa: ANN001, ANN003 - urlopen shape
    """A ``/v1/models`` payload, so the bench health check passes."""

    class _Resp:
        def read(self) -> bytes:
            return b'{"data": [{"id": "stub-model"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ANN002 - context manager
            return False

    return _Resp()


def _stub_stream_chat(  # noqa: ANN001, ANN003 - _stream_chat shape
    base_url, model, prompt, max_tokens=1024, api_key=None
):
    """A completed benchmark request, so a level sweeps without a server."""
    from imas_ambix.agent.bench import BenchResult

    return BenchResult(
        category="concurrency",
        completion_tokens=10,
        prompt_tokens=5,
        status="passed",
        model=model,
    )
