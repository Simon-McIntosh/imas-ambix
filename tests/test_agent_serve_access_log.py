"""Launch-line contracts for suppressing polled request paths from the serve log.

A serve's uvicorn access log records every request, and the paths polled on a
fixed cadence by watchers, health checks and catalog discovery dominate it:
measured on the four-card SGLang serve, ``/v1/models`` and ``/metrics`` were
60% of 2,719,458 lines, crowding out the engine's own prefill and decode lines.

Each engine expresses the exclusion differently -- SGLang takes the prefixes as
one space-separated option group, vLLM as a single comma-separated value -- so
the contract under test is the generated launch line rather than a profile
attribute: a flag set on a profile that never reaches the engine is exactly the
failure this guards. Where the engine's own help spells an option, the emitted
flag must be the one the help spells, never a spelling carried here from
memory; an engine whose help spells none must get none.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import _ACCESS_LOG_EXCLUDE_PREFIXES, generate_serve_script

# The paths the change excludes, pinned literally so the test cannot be made to
# pass by shrinking the set the code and the test share.
_EXPECTED_PREFIXES = ("/metrics", "/v1/models", "/health")

# Request paths whose lines must survive: an excluded prefix that swallowed
# these would suppress exactly the traffic the log is read for.
_GENERATION_PATHS = ("/v1/messages", "/v1/chat/completions")

# Engine -> the option its help spells for excluding request paths from the
# access log. An engine absent from this map has no such option and its
# generated script must carry none.
_EXCLUSION_FLAG = {
    "sglang": "--uvicorn-access-log-exclude-prefixes",
    "vllm": "--disable-access-log-for-endpoints",
}

_LAUNCH_MARKER = {
    "sglang": "sglang.launch_server",
    "vllm": "vllm.entrypoints.openai.api_server",
}

# vLLM builds its argument parser through a device probe that fails without a
# GPU, so its help is unreachable on a login node unless a target device is
# named. The option list is a property of the installed wheel, not of the
# hardware, so naming a device that needs no GPU is what makes it readable.
_HELP_ENVIRONMENT = {"vllm": {"VLLM_TARGET_DEVICE": "cpu"}}

_OPTION = re.compile(r"--[a-z0-9][a-z0-9-]*")

_PROFILE_BY_ENGINE = {"sglang": "deepseek-v4-1-flash", "vllm": "deepseek-v4-flash"}


def _launch_line(script: str, engine_type: str) -> str:
    return next(
        line for line in script.splitlines() if _LAUNCH_MARKER[engine_type] in line
    )


def _excluded_paths(tokens: list[str], flag: str) -> list[str]:
    """Paths *flag* excludes on a launch line, however the engine encodes them.

    SGLang gives each prefix as its own argument; vLLM gives one comma-joined
    to the option list, so the value is split before comparing.
    """
    if flag not in tokens:
        return []
    values: list[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return [path for value in values for path in value.split(",") if path]


def _help_text(engine_type: str) -> str | None:
    python = SiteConfig().python_path(engine_type)
    if not Path(python).is_file():
        return None
    completed = subprocess.run(
        [str(python), "-m", _LAUNCH_MARKER[engine_type], "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, **_HELP_ENVIRONMENT.get(engine_type, {})},
        timeout=300,
    )
    text = completed.stdout + completed.stderr
    # Help that imported but printed no option lines is an instrument failure,
    # not an engine fact: a reader that sees nothing must prove it can see
    # something before its absence is reported as a missing option.
    return text if _OPTION.search(text) else None


def test_serve_line_excludes_every_polled_path() -> None:
    """The four-card SGLang serve drops /metrics, /v1/models and /health."""
    launch = _launch_line(
        generate_serve_script(
            load_profile("deepseek-v4-1-flash"), SiteConfig(), port=18810
        ),
        "sglang",
    )

    assert "--uvicorn-access-log-exclude-prefixes" in launch
    assert _excluded_paths(
        shlex.split(launch), "--uvicorn-access-log-exclude-prefixes"
    ) == list(_EXPECTED_PREFIXES)


def test_request_paths_are_not_swallowed_by_an_excluded_prefix() -> None:
    """Generation request lines stay in the log."""
    assert not [
        path
        for path in _GENERATION_PATHS
        for prefix in _EXPECTED_PREFIXES
        if path.startswith(prefix)
    ]


def test_exclusion_prefixes_are_metrics_models_and_health() -> None:
    assert tuple(_ACCESS_LOG_EXCLUDE_PREFIXES) == _EXPECTED_PREFIXES


@pytest.mark.parametrize("engine_type", ["sglang", "vllm"])
def test_serve_flag_is_emitted_only_where_the_engine_help_lists_one(
    engine_type: str,
) -> None:
    """The generated flag tracks the installed engine's own option list."""
    help_text = _help_text(engine_type)
    if help_text is None:
        pytest.skip(f"{engine_type} help is not producible on this host")

    flag = _EXCLUSION_FLAG[engine_type]
    launch = _launch_line(
        generate_serve_script(
            load_profile(_PROFILE_BY_ENGINE[engine_type]), SiteConfig(), port=18800
        ),
        engine_type,
    )
    tokens = shlex.split(launch)

    if flag in help_text:
        assert flag in tokens, f"{engine_type} help lists {flag} but the serve omits it"
        assert _excluded_paths(tokens, flag) == list(_EXPECTED_PREFIXES)
    else:
        assert flag not in tokens


def test_sglang_help_spells_the_flag_the_script_emits() -> None:
    """The spelling is read from the environment, never carried in this repo."""
    help_text = _help_text("sglang")
    if help_text is None:
        pytest.skip("sglang help is not producible on this host")

    assert "--uvicorn-access-log-exclude-prefixes" in help_text
