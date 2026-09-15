"""The prefix probe must locate where reuse dies, and log no prompt content."""
from __future__ import annotations

import logging

from imas_ambix.agent.router import RouterApp


class _R:
    async def resolve(self): return []


def _scope():
    return {
        "headers": [(b"user-agent", b"claude-cli/2.1")],
        "client": ("10.0.0.1", 5150),
    }


def _turns(monkeypatch, caplog, first, second):
    monkeypatch.setenv("AMBIX_ROUTER_PREFIX_PROBE", "1")
    app = RouterApp(_R())
    with caplog.at_level(logging.INFO):
        app._log_prefix_divergence({"messages": first}, _scope())
        app._log_prefix_divergence({"messages": second}, _scope())
    return [r.getMessage() for r in caplog.records if "prefix-probe" in r.getMessage()]


def test_a_stable_head_keeps_its_shallow_digest(monkeypatch, caplog):
    """Turn N+1 extending turn N must match at every depth it shares."""
    base = [
        {"role": "system", "content": "S" * 4000},
        {"role": "user", "content": "one"},
    ]
    grown = base + [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "two"},
    ]
    a, b = _turns(monkeypatch, caplog, base, grown)
    assert a.split()[3] == b.split()[3], "shallow digest must survive an append"


def test_a_varying_head_diverges_at_the_shallowest_depth(monkeypatch, caplog):
    """One changed byte at the top invalidates everything after it."""
    first = [{"role": "system", "content": "run-id-AAAA " + "S" * 4000}]
    second = [{"role": "system", "content": "run-id-BBBB " + "S" * 4000}]
    a, b = _turns(monkeypatch, caplog, first, second)
    assert a.split()[3] != b.split()[3], "a moving head must show at depth 512"


def test_the_probe_logs_no_prompt_content(monkeypatch, caplog):
    secret = "CANARY-e7f1a9-DO-NOT-LOG"
    lines = _turns(
        monkeypatch,
        caplog,
        [{"role": "user", "content": secret}],
        [{"role": "user", "content": secret}],
    )
    assert lines and all(secret not in line for line in lines)


def test_it_is_off_unless_asked(monkeypatch, caplog):
    monkeypatch.delenv("AMBIX_ROUTER_PREFIX_PROBE", raising=False)
    app = RouterApp(_R())
    with caplog.at_level(logging.INFO):
        app._log_prefix_divergence(
            {"messages": [{"role": "user", "content": "x"}]}, _scope()
        )
    assert not [r for r in caplog.records if "prefix-probe" in r.getMessage()]
