"""Benchmark utilities for ``ambix agent bench``."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class BenchResult:
    """Metrics from a single benchmark request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_s: float = 0.0
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    model: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class BenchSuite:
    """Aggregate results from a benchmark suite."""

    results: list[BenchResult] = field(default_factory=list)
    model: str = ""

    @property
    def successful(self) -> list[BenchResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[BenchResult]:
        return [r for r in self.results if not r.ok]

    def summary(self) -> dict[str, object]:
        ok = self.successful
        if not ok:
            return {"total": len(self.results), "passed": 0, "failed": len(self.results)}
        tps_values = [r.tokens_per_second for r in ok]
        ttft_values = [r.time_to_first_token_s for r in ok]
        total_tokens = sum(r.completion_tokens for r in ok)
        total_time = sum(r.total_time_s for r in ok)
        return {
            "model": self.model,
            "total": len(self.results),
            "passed": len(ok),
            "failed": len(self.failed),
            "total_tokens": total_tokens,
            "total_time_s": round(total_time, 2),
            "avg_tps": round(sum(tps_values) / len(tps_values), 1),
            "min_tps": round(min(tps_values), 1),
            "max_tps": round(max(tps_values), 1),
            "avg_ttft_ms": round(sum(ttft_values) / len(ttft_values) * 1000, 1),
            "min_ttft_ms": round(min(ttft_values) * 1000, 1),
            "max_ttft_ms": round(max(ttft_values) * 1000, 1),
        }


# ── Benchmark probes ────────────────────────────────────────────────


def _stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.6,
) -> BenchResult:
    """Send a streaming chat completion and measure timing."""
    import urllib.error
    import urllib.request

    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    result = BenchResult(model=model)
    t0 = time.perf_counter()
    first_token_time: float | None = None
    completion_text = []
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            buffer = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line or line == b"data: [DONE]":
                        continue
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            completion_text.append(content)
                            token_count += 1
                        usage = data.get("usage")
                        if usage:
                            result.prompt_tokens = usage.get(
                                "prompt_tokens", 0
                            )
                            result.completion_tokens = usage.get(
                                "completion_tokens", token_count
                            )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.error = str(exc)
        result.total_time_s = time.perf_counter() - t0
        return result

    t_end = time.perf_counter()
    result.total_time_s = t_end - t0
    if first_token_time is not None:
        result.time_to_first_token_s = first_token_time - t0
    if result.completion_tokens == 0:
        result.completion_tokens = token_count
    if result.completion_tokens > 0 and result.total_time_s > 0:
        gen_time = t_end - (first_token_time or t0)
        if gen_time > 0:
            result.tokens_per_second = result.completion_tokens / gen_time
    return result


# ── Preset prompts ──────────────────────────────────────────────────

BENCH_PRESETS: dict[str, dict[str, object]] = {
    "short": {
        "description": "Short generation — quick TPS measurement",
        "messages": [
            {"role": "user", "content": "What is nuclear fusion? Reply in exactly 3 sentences."}
        ],
        "max_tokens": 128,
    },
    "medium": {
        "description": "Medium generation — sustained throughput",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the tokamak concept for magnetic confinement fusion. "
                    "Cover the key physics principles, the role of the toroidal and "
                    "poloidal magnetic fields, and why this approach is promising "
                    "for energy production. Aim for about 400 words."
                ),
            }
        ],
        "max_tokens": 512,
    },
    "long": {
        "description": "Long generation — sustained decode speed",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed technical guide on implementing an OpenAI-compatible "
                    "API server for serving large language models. Cover architecture, "
                    "tensor parallelism, KV cache management, continuous batching, "
                    "and deployment best practices. Include code examples."
                ),
            }
        ],
        "max_tokens": 2048,
    },
    "code": {
        "description": "Code generation — practical coding task",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function that implements a binary search tree with "
                    "insert, search, delete, and in-order traversal. Include type hints "
                    "and docstrings. Then write pytest tests for all operations."
                ),
            }
        ],
        "max_tokens": 1024,
    },
    "thinking": {
        "description": "Reasoning — deep thinking / chain-of-thought",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A tokamak plasma has a major radius R=6.2m, minor radius a=2.0m, "
                    "toroidal field B_T=5.3T, and plasma current I_p=15MA. "
                    "Estimate the safety factor q at the plasma edge using the "
                    "cylindrical approximation. Then explain what happens physically "
                    "if q_edge drops below 2, and why this matters for ITER."
                ),
            }
        ],
        "max_tokens": 1024,
    },
    "tool_use": {
        "description": "Tool calling — test function call capability",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in Cadarache, France right now?",
            }
        ],
        "max_tokens": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name, e.g. Cadarache, France",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    },
}


def run_bench_preset(
    base_url: str,
    model: str,
    preset: str,
    *,
    repeat: int = 1,
) -> BenchSuite:
    """Run a benchmark preset ``repeat`` times and return aggregated results."""
    cfg = BENCH_PRESETS[preset]
    suite = BenchSuite(model=model)
    for _ in range(repeat):
        result = _stream_chat(
            base_url=base_url,
            model=model,
            messages=cfg["messages"],  # type: ignore[arg-type]
            max_tokens=cfg.get("max_tokens", 512),  # type: ignore[arg-type]
        )
        suite.results.append(result)
    return suite


def run_tool_call_bench(
    base_url: str,
    model: str,
) -> BenchResult:
    """Test tool-call capability with a single request (non-streaming)."""
    import urllib.error
    import urllib.request

    cfg = BENCH_PRESETS["tool_use"]
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": cfg["messages"],
            "tools": cfg["tools"],
            "max_tokens": cfg.get("max_tokens", 256),
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    result = BenchResult(model=model)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.error = str(exc)
        result.total_time_s = time.perf_counter() - t0
        return result

    result.total_time_s = time.perf_counter() - t0
    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            result.completion_tokens = data.get("usage", {}).get(
                "completion_tokens", 0
            )
            result.prompt_tokens = data.get("usage", {}).get(
                "prompt_tokens", 0
            )
        else:
            result.error = f"No tool calls in response: {msg.get('content', '')[:200]}"
    else:
        result.error = "Empty response"
    return result
