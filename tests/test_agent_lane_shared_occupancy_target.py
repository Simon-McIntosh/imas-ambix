"""The lane estimator and the router's memory rule read one occupancy target.

One pool described by two targets is a figure no reader can act on. The
estimator publishes a capacity from its share of the pool; the router's memory
rule sizes the gate's automatic width from its own. If those two shares differ,
the lane document and the gate describe the same engine differently and neither
records which one it used, so a coordinator sizing a wave off the published
headroom can dispatch past the width the gate will admit.

The target is therefore one object both read. Identity is asserted by ``is``
rather than by comparing two equal literals: a pair of 0.90s is exactly what a
later re-tuning leaves behind, and two equal literals cannot catch it. The value
itself is pinned once, in the same file, so the identity cannot be satisfied by a
shared object that has quietly moved.
"""

from __future__ import annotations

import importlib
import json

from imas_ambix.agent import lane, router
from imas_ambix.agent.lane import LaneCapacity, read_occupancy_target
from imas_ambix.agent.router import RouterApp


class _NoUpstreams:
    """A resolver naming no engine: this file exercises composition, not serving."""

    async def resolve(self):
        return ()


def _capacity() -> LaneCapacity:
    """The done-when reading: pool 4M at 18% with five running, mean context 144k."""
    return LaneCapacity(
        model_id="deepseek-v4-flash",
        pool_tokens=4_000_000,
        running=5,
        waiting=0,
        kv_occupancy=0.18,
        preemptions=0,
        prefix_hit_rate=None,
    )


def test_the_estimator_and_the_memory_rule_read_one_object():
    """Identity, not two equal literals, and the one value pinned here."""
    assert router.DEFAULT_AUTO_OCCUPANCY_TARGET is lane.DEFAULT_OCCUPANCY_TARGET
    assert LaneCapacity.OCCUPANCY_TARGET is lane.DEFAULT_OCCUPANCY_TARGET
    assert lane.DEFAULT_OCCUPANCY_TARGET == 0.90


def test_the_gates_own_default_settings_carry_the_shared_object():
    """The memory rule reads ``settings.occupancy_target``, so it must be that object.

    Asserting only the module alias would leave the wiring between the alias and
    the field the memory rule actually divides by unowned: a gate built with its
    documented defaults is where the running router reads it. With no control
    file the gate answers from those defaults, so this is the value in force on a
    router launched without the automatic width switched on.
    """
    gate = router._GenerationGate(None)

    assert gate.settings().occupancy_target is lane.DEFAULT_OCCUPANCY_TARGET


def test_the_shared_target_sizes_the_pool_arithmetic():
    """4,000,000 x 0.90 // 144,000 is 25, and 25 less five running is 20 spare."""
    capacity = _capacity()

    assert capacity.mean_context == 144_000
    assert capacity.concurrent_requests == 25
    assert capacity.headroom == 20


def test_the_published_headroom_is_the_smaller_of_memory_and_the_gate(tmp_path):
    """At width 16 with five in flight the gate has 11, below memory's 20.

    A document leads with one headroom an orchestrator sizes from, so it must be
    the binding figure. Here the pool arithmetic would allow twenty more requests
    while the admitted width allows eleven; publishing memory's larger number
    would invite a dispatch the gate then queues. The memory figure stays visible
    beside it, so a reader can still see which constraint bound.
    """
    lane_path = tmp_path / "lane.json"
    lane.write_lane_document(_capacity(), lane_path)

    gate_path = tmp_path / "router-gate.json"
    gate_path.write_text(json.dumps({"width": 16}), encoding="utf-8")

    app = RouterApp(_NoUpstreams(), lane_document=lane_path, gate_file=gate_path)
    app._generation_gate._in_flight = 5
    app._publish_gate_snapshot()

    document = json.loads(lane_path.read_text(encoding="utf-8"))
    assert document["admission"]["verdict"] == "open"
    assert document["admission"]["headroom"] == 11
    assert document["engine_headroom"] == 20
    assert document["headroom"] == 11


def test_the_environment_override_still_wins():
    """The one constant is overridable, and the override is not the default."""
    override = read_occupancy_target({lane.OCCUPANCY_TARGET_ENV: "0.75"})

    assert override == 0.75
    assert override != lane.DEFAULT_OCCUPANCY_TARGET


def test_the_override_moves_the_memory_rule_default_with_the_estimator(monkeypatch):
    """An override must move the router's memory-rule default, not only the
    estimator's published capacity.

    The estimator resolves ``IMAS_AMBIX_LANE_OCCUPANCY_TARGET`` at import into
    ``LaneCapacity.OCCUPANCY_TARGET``; the memory rule's default must follow that
    resolved figure. Binding it to the unresolved module constant instead leaves
    an override that moves the published capacity while the gate keeps sizing the
    pool at the default -- the two-targets-on-one-pool disagreement this file
    exists to prevent. Both modules are re-imported under the override, and the
    environment is restored afterwards so no later reader sees it.
    """
    monkeypatch.setenv(lane.OCCUPANCY_TARGET_ENV, "0.7")
    try:
        importlib.reload(lane)
        importlib.reload(router)

        estimator_target = lane.LaneCapacity.OCCUPANCY_TARGET
        memory_rule_default = router.DEFAULT_AUTO_OCCUPANCY_TARGET

        assert estimator_target == 0.7
        assert memory_rule_default == 0.7
        assert memory_rule_default is estimator_target
        assert router._GenerationGate(None).settings().occupancy_target == 0.7
    finally:
        monkeypatch.delenv(lane.OCCUPANCY_TARGET_ENV, raising=False)
        importlib.reload(lane)
        importlib.reload(router)
