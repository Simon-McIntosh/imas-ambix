"""The lane reader sums a cumulative counter over all of its label series.

An engine may publish one metric under several labels: SGLang labels the
generation counter with ``is_streaming`` and reports the streaming series as
the live one, and vLLM repeats a counter on each engine rank. A counter is a
running total, so every label set contributes to it. Reading only the first
series reports one shard as the whole, which for the generation counter is the
near-frozen non-streaming shard -- the value that made the width controller see
zero tokens per second against a climbing engine. A gauge instead repeats one
instantaneous reading, so its label sets must not be added.
"""

from __future__ import annotations

from imas_ambix.agent.lane import parse_lane_capacity

_MODEL = "deepseek-v4.1-flash"

#: The SGLang generation counter as the live engine publishes it: a small
#: non-streaming shard and the large streaming series that carries the traffic.
_GENERATION_NON_STREAMING = 408.0
_GENERATION_STREAMING = 4_500_000.0
_GENERATION_TOTAL = _GENERATION_NON_STREAMING + _GENERATION_STREAMING


def _sglang_pool() -> str:
    return f'sglang:max_total_num_tokens{{model_name="{_MODEL}",tp_rank="0"}} 4000000.0'


def _generation_line(streaming: str, value: float) -> str:
    return (
        "sglang:generation_tokens_total"
        f'{{is_streaming="{streaming}",model_name="{_MODEL}",tp_rank="0"}} {value}'
    )


def _sglang_body(*lines: str) -> str:
    return "\n".join((_sglang_pool(), *lines))


def test_generation_tokens_sums_every_series():
    """The two generation series yield their sum, not the first of them."""
    body = _sglang_body(
        _generation_line("false", _GENERATION_NON_STREAMING),
        _generation_line("true", _GENERATION_STREAMING),
    )

    capacity = parse_lane_capacity(body)

    assert capacity.generation_tokens == _GENERATION_TOTAL


def test_single_series_counter_is_unchanged():
    """One series summing to itself is the plain reading, not a doubled one."""
    body = _sglang_body(_generation_line("false", 104.0))

    capacity = parse_lane_capacity(body)

    assert capacity.generation_tokens == 104


def test_single_series_gauge_is_unchanged():
    """A gauge read as one value keeps that value."""
    occupancy = f'sglang:full_token_usage{{model_name="{_MODEL}",tp_rank="0"}} 0.25'
    body = _sglang_body(_generation_line("false", 104.0), occupancy)

    capacity = parse_lane_capacity(body)

    assert capacity.kv_occupancy == 0.25


def test_gauge_repeated_on_each_rank_keeps_one_reading():
    """A gauge on two vLLM engine ranks is one reading, so it is not summed."""
    labels = 'model_name="vllm-model"'
    body = "\n".join(
        (
            f'vllm:num_requests_running{{{labels},engine="0"}} 3.0',
            f'vllm:num_requests_running{{{labels},engine="1"}} 3.0',
            f'vllm:cache_config_info{{{labels},kv_cache_size_tokens="123456"}} 1.0',
        )
    )

    capacity = parse_lane_capacity(body)

    assert capacity.running == 3


def test_cumulative_lane_roles_sum_over_every_series():
    """Preemptions and the external-cache counters sum across engine ranks."""
    labels = 'model_name="vllm-model"'
    body = "\n".join(
        (
            f'vllm:num_requests_running{{{labels},engine="0"}} 1.0',
            f'vllm:cache_config_info{{{labels},kv_cache_size_tokens="123456"}} 1.0',
            f'vllm:num_preemptions_total{{{labels},engine="0"}} 4.0',
            f'vllm:num_preemptions_total{{{labels},engine="1"}} 6.0',
            f'vllm:external_prefix_cache_queries_total{{{labels},engine="0"}} 100.0',
            f'vllm:external_prefix_cache_queries_total{{{labels},engine="1"}} 50.0',
            f'vllm:external_prefix_cache_hits_total{{{labels},engine="0"}} 30.0',
            f'vllm:external_prefix_cache_hits_total{{{labels},engine="1"}} 20.0',
        )
    )

    capacity = parse_lane_capacity(body)

    assert capacity.preemptions == 10
    assert capacity.external_hit_rate == 50 / 150
