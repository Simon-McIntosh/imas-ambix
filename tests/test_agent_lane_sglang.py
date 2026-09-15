"""SGLang metrics retain the lane reader's engine-independent contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.agent.lane import (
    LaneCapacity,
    parse_lane_capacity,
    write_lane_document,
)

_FIXTURE = Path(__file__).parent / "data" / "sglang_metrics_sample.txt"


def test_sglang_metrics_produce_a_populated_lane_capacity(tmp_path):
    """The live receipt contains one rank-zero device pool and host cache tier."""
    capacity = parse_lane_capacity(_FIXTURE.read_text(encoding="utf-8"))

    assert capacity == LaneCapacity(
        model_id="deepseek-v4.1-flash",
        pool_tokens=4_000_000,
        running=0,
        waiting=0,
        kv_occupancy=0.0,
        preemptions=None,
        prefix_hit_rate=0.0,
        hicache_host_total_tokens=8_000_000,
        hicache_host_used_tokens=55_552,
    )
    assert capacity.binding_observed is None
    document = json.loads(
        write_lane_document(capacity, tmp_path / "lane.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["hicache_host"] == {
        "total_tokens": 8_000_000,
        "used_tokens": 55_552,
        "used_fraction": 0.006944,
    }


def test_vllm_metrics_keep_the_existing_capacity_values():
    """Engine dispatch preserves the vLLM parser's established output."""
    labels = 'engine="0",model_name="vllm-model"'
    body = "\n".join(
        (
            f"vllm:num_requests_running{{{labels}}} 3.0",
            f"vllm:num_requests_waiting{{{labels}}} 2.0",
            f"vllm:kv_cache_usage_perc{{{labels}}} 0.4",
            f"vllm:num_preemptions_total{{{labels}}} 5.0",
            f"vllm:prefix_cache_queries_total{{{labels}}} 40.0",
            f"vllm:prefix_cache_hits_total{{{labels}}} 10.0",
            f'vllm:cache_config_info{{{labels},kv_cache_size_tokens="123456"}} 1.0',
        )
    )

    assert parse_lane_capacity(body) == LaneCapacity(
        model_id="vllm-model",
        pool_tokens=123_456,
        running=3,
        waiting=2,
        kv_occupancy=0.4,
        preemptions=5,
        prefix_hit_rate=0.25,
    )


def test_metrics_with_neither_engine_family_keep_the_existing_refusal():
    """A document without an engine pool cannot become an invented capacity."""
    with pytest.raises(ValueError, match="engine metrics carry no KV pool size"):
        parse_lane_capacity('process_cpu_seconds_total{component="reader"} 1.0')
