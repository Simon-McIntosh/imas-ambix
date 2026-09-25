"""ASGI routing for native model-serving protocols and bounded generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp

from imas_ambix.agent.request_receipts import (
    DEFAULT_MAX_ROWS_PER_S,
    DEFAULT_WINDOW_S,
    RECEIPTS_FILENAME,
    SELF_ANSWERED_UPSTREAM,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    RequestReceiptSink,
    StreamAccounting,
)
from imas_ambix.agent.vllm_catalog import validate_catalog_metadata

AsgiMessage = dict[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]
if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# The receipt sampling ceiling is a bound, and a bound that lives only in an
# invocation is one a launch eventually omits -- after which the value in force
# is a source default nobody reasoned about and nothing reports. Naming it in
# the environment lets a full-fidelity campaign raise it or remove it
# (``inf`` keeps every row) without editing source, and the value in force is
# logged when the sink is built. A value that is not a positive number is a
# launch error rather than a silent fallback to the default: a record sampled at
# a rate the operator did not choose reads as complete and is not.
RECEIPT_MAX_ROWS_PER_S_ENV = "AMBIX_ROUTER_RECEIPT_MAX_ROWS_PER_S"
RECEIPT_WINDOW_S_ENV = "AMBIX_ROUTER_RECEIPT_WINDOW_S"

GATE_FILENAME = "router-gate.json"
DEFAULT_GENERATION_WIDTH = 22
DEFAULT_GENERATION_WAIT_SECONDS = 300.0
GENERATION_RETRY_AFTER_SECONDS = 5
_GATE_CONFIG_REFRESH_SECONDS = 1.0
# A pause can end the relays already in flight instead of waiting for the
# longest turn to finish. The cut is a property of a pause, never a mode on its
# own: cutting with no pause would end each relay at the moment it was admitted,
# so ``cut`` is honoured only while ``paused`` holds. The two forms answer the
# cut client stream differently -- ``close`` drops the connection with no
# terminal chunk, which a streaming client retries as a streaming request, and the
# ``error`` form writes an SSE overloaded_error frame and ends the stream for
# clients that need an explicit signal. Both cancel the upstream request, so the
# engine aborts it rather than finishing a generation nobody will read.
GATE_CUT_FORMS = ("close", "error")
DEFAULT_GATE_CUT_FORM = "close"
# The cut is discovered by polling the same gate file the rest of the settings
# come from, so an operator edit reaches a relay already streaming without
# anyone holding a handle to it. This interval bounds how long an in-flight
# relay can survive a cut that has already been written.
_CUT_POLL_SECONDS = 0.2


def _cut_error_frame() -> bytes:
    """One Anthropic SSE error event carrying an overloaded_error."""
    payload = {
        "type": "error",
        "error": {
            "type": "overloaded_error",
            "message": "generation cut by an operator pause; retry the request",
        },
    }
    return b"event: error\ndata: " + json.dumps(payload).encode() + b"\n\n"

# The automatic width mode sizes the gate to the pool the engine actually holds
# rather than to a number an operator picks once and forgets. The width is
# floor(pool tokens x target / context), so it moves with the workload: a lane
# whose sessions hold small contexts admits more of them, and one carrying large
# contexts admits fewer. The floor is the width below which the lane would stop
# serving the sessions already waiting on it; the cap is the engine's own
# running ceiling, which the gate must never advertise past because the engine
# cannot admit more than it regardless.
GATE_AUTO_WIDTH = "auto"
DEFAULT_AUTO_OCCUPANCY_TARGET = 0.90
DEFAULT_AUTO_WIDTH_FLOOR = 16
DEFAULT_AUTO_WIDTH_CAP = 36
# The context estimate is deliberately slow: one reading that happens to catch
# an unusual working set must not swing the admitted width. A time constant of
# ten minutes at the thirty-second lane cadence weights each new sample at under
# five percent, so a single outlier moves the estimate by well under a tenth of
# its value while a genuine shift still arrives within a few minutes.
DEFAULT_AUTO_CONTEXT_TIME_CONSTANT_SECONDS = 600.0

# The pool arithmetic above answers "how many contexts fit", which is a
# different question from "how many are worth admitting". The two disagree
# because the prefix cache and the per-stream decode rate fall before the pool
# is full: measured on this lane the aggregate throughput plateaus from about
# twelve running while the pool sits at half occupancy and preemptions stay at
# zero, so a width chosen for pool population admits every stream past the point
# where they all slow down still costs every stream already running and buys no
# aggregate throughput. The controller here searches the lane's own measured
# throughput for the smallest width that holds the plateau. It keeps the pool
# arithmetic as the ceiling it may not exceed -- a width the pool cannot hold is
# wrong however fast it looks -- and as the width before it has any measurement
# to reason from.
GATE_CONTROLLER_STEP = 2
# Scrapes per decision. One scrape of this lane carries roughly sixty tokens per
# second of noise against a rise of a few hundred, so a single scrape cannot
# separate a real step from a draw; sixteen average to a noise near fifteen, and
# a step of that size then clears the threshold many times over. The cost is
# latency rather than correctness: a decision takes several lane intervals, so
# the controller settles in tens of minutes and not seconds.
GATE_CONTROLLER_WINDOWS = 16
GATE_CONTROLLER_NOISE_TOK_S = 60.0
# The throughput change a judged step must show before the controller believes
# it. It is a floor on what counts as evidence rather than a target: above the
# noise the judged average carries, which is what stops the upward walk from
# climbing a plateau that is flat, and below a genuine step, which is what stops
# it discarding one. Derived from the noise and the number of scrapes behind each
# judgement so the number carries its own provenance.
GATE_CONTROLLER_MATERIAL_TOK_S = (
    3.2 * GATE_CONTROLLER_NOISE_TOK_S / math.sqrt(GATE_CONTROLLER_WINDOWS)
)
# Aggregate throughput is not the only thing a width can ruin. A lane that is
# fast in total while every stream crawls is one nobody can use, and the prefix
# cache -- which is what pays for the context every stream shares -- is evicted
# by width long before the pool fills. These two read the lane's health and step
# the width down directly, ahead of the throughput objective, because a reading
# taken while the cache is thrashing describes the damage rather than the width.
GATE_CONTROLLER_PER_STREAM_FLOOR_TOK_S = 10.0
GATE_CONTROLLER_PREFIX_HIT_FLOOR = 0.95
GATE_CONTROLLER_KV_CEILING = 0.6
# The narrowest width the search may resolve. The pool arithmetic's own floor is
# a different kind of number -- it says how many contexts the engine can hold,
# and its default sits above the measured knee -- so it must not also bound a
# search whose whole purpose is to find a width below it. An operator floor lower
# than this one is still honoured, which is why the controller takes the smaller
# of the two.
GATE_CONTROLLER_WIDTH_FLOOR = 4


@dataclass(frozen=True, slots=True)
class _GateSettings:
    width: int
    wait_seconds: float
    auto: bool = False
    occupancy_target: float = DEFAULT_AUTO_OCCUPANCY_TARGET
    width_floor: int = DEFAULT_AUTO_WIDTH_FLOOR
    width_cap: int = DEFAULT_AUTO_WIDTH_CAP
    # A pause outranks the width: an operator declaring the lane paused wants no
    # NEW generation to start, whatever the gate would otherwise admit. Requests
    # already admitted run to completion unless ``cut`` is also set, in which
    # case they are ended instead and each cut client stream is answered by
    # ``cut_form``: an overloaded_error frame, or the connection dropped without
    # a terminal chunk. The upstream request is cancelled in both forms, and
    # callers that arrive during the pause wait in the same FIFO as any other
    # caller, so clearing the flag admits them in arrival order.
    paused: bool = False
    cut: bool = False
    cut_form: str = DEFAULT_GATE_CUT_FORM
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Admission:
    outcome: str
    disconnect_task: asyncio.Task[None] | None = None
    retry_after_seconds: int = 1
    gate_wait_s: float = 0.0


class _GenerationGate:
    """Admit post-catalog generation relays in FIFO order up to a live width."""

    def __init__(
        self,
        config_path: Path | None,
        *,
        default_width: int = DEFAULT_GENERATION_WIDTH,
        default_wait_seconds: float = DEFAULT_GENERATION_WAIT_SECONDS,
    ) -> None:
        self.config_path = config_path
        self._defaults = _GateSettings(default_width, default_wait_seconds)
        self._cached_settings = self._defaults
        self._last_integer_width: int | None = None
        self._auto_transition_pending = False
        self._last_logged_settings: _GateSettings | None = None
        self._next_config_refresh_at = 0.0
        self._condition = asyncio.Condition()
        self._waiters: deque[tuple[object, float]] = deque()
        self._in_flight = 0
        # Automatic width state, fed only by fresh measured readings. The
        # estimate starts undefined so the first reading seeds it whole rather
        # than being averaged against a guess, and the last computed width is
        # retained so a lane that goes quiet holds the width it last justified
        # instead of dropping to nothing.
        self._context_estimate: float | None = None
        self._context_stamp = 0.0
        self._pool_tokens: int | None = None
        self._effective_width: int | None = None
        # The width controller's state, fed by the same fresh readings. The
        # bounds are refreshed on every resolve rather than configured here, so
        # an operator edit to the gate file moves the ceiling without a restart.
        self._controller_width: int | None = None
        self._controller_lower: int | None = None
        self._controller_upper: int | None = None
        self._controller_phase = "climb"
        self._controller_proposal: int | None = None
        self._controller_reference_width: int | None = None
        self._controller_reference_rate: float | None = None
        self._controller_best_rate: float | None = None
        self._controller_rates: list[float] = []
        self._controller_health: list[
            tuple[float | None, float | None, float | None]
        ] = []
        self._controller_note: str | None = "memory rule: no throughput reading"
        self._throughput_tokens: int | None = None
        self._throughput_stamp = 0.0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    def admission_snapshot(self) -> dict[str, object]:
        """Return the queue's demand signal independently of engine capacity."""
        settings = self.settings()
        if self._waiters:
            oldest_wait_seconds: float | None = max(
                0.0, time.monotonic() - self._waiters[0][1]
            )
        else:
            oldest_wait_seconds = None
        headroom = settings.width - self.in_flight - self.waiting
        if settings.paused:
            headroom = min(headroom, 0)
            verdict = "paused"
        elif self.waiting > 0:
            verdict = "congested"
        elif self.in_flight >= settings.width:
            verdict = "full"
        else:
            verdict = "open"
        return {
            "headroom": headroom,
            "oldest_wait_seconds": oldest_wait_seconds,
            "waiting": self.waiting,
            "verdict": verdict,
        }

    def settings(self) -> _GateSettings:
        """Read changed configuration and otherwise return the cached settings."""
        if self.config_path is None:
            return self._defaults
        now = time.monotonic()
        if now < self._next_config_refresh_at:
            return self._cached_settings
        self._next_config_refresh_at = now + _GATE_CONFIG_REFRESH_SECONDS
        try:
            self.config_path.stat()
        except OSError:
            available = False
        else:
            available = True

        settings = self._defaults
        if available:
            try:
                payload = json.loads(self.config_path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("the root must be an object")
                self._auto_transition_pending = (
                    payload.get("width") == GATE_AUTO_WIDTH
                    and not self._cached_settings.auto
                    and self._last_integer_width is not None
                )
                settings = self._settings_from_payload(payload)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                logger.warning(
                    "generation gate config ignored path=%s error=%s: %s; "
                    "using defaults width=%d wait_seconds=%s",
                    self.config_path,
                    type(error).__name__,
                    error,
                    self._defaults.width,
                    self._defaults.wait_seconds,
                )
        if not settings.auto:
            self._last_integer_width = settings.width
        self._auto_transition_pending = False
        self._cached_settings = settings
        if settings != self._last_logged_settings:
            logger.info(
                "generation gate config path=%s width=%d wait_seconds=%s "
                "paused=%s reason=%s",
                self.config_path,
                settings.width,
                settings.wait_seconds,
                settings.paused,
                settings.reason,
            )
            self._last_logged_settings = settings
        return settings

    def _settings_from_payload(self, payload: Mapping[str, Any]) -> _GateSettings:
        """Resolve one gate file into settings, including the automatic width.

        ``width`` accepts the literal ``"auto"`` in place of an integer. Every
        other key is optional and a malformed one is an error rather than a
        quiet fallback, so a typo in ``occupancy_target`` cannot leave the gate
        sizing against a number nobody chose.
        """
        wait_seconds = payload.get("wait_seconds", self._defaults.wait_seconds)
        if (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, int | float)
            or not math.isfinite(wait_seconds)
            or wait_seconds <= 0
        ):
            raise ValueError("wait_seconds must be a finite positive number")
        paused = payload.get("paused", False)
        if type(paused) is not bool:
            raise ValueError("paused must be a boolean")
        cut = payload.get("cut", False)
        if type(cut) is not bool:
            raise ValueError("cut must be a boolean")
        cut_form = payload.get("cut_form", DEFAULT_GATE_CUT_FORM)
        if cut_form not in GATE_CUT_FORMS:
            raise ValueError('cut_form must be one of "close" or "error"')
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string")
        target = payload.get("occupancy_target", DEFAULT_AUTO_OCCUPANCY_TARGET)
        raw_width = payload.get("width", self._defaults.width)
        if raw_width == GATE_AUTO_WIDTH:
            floor = payload.get("width_floor", DEFAULT_AUTO_WIDTH_FLOOR)
            cap = payload.get("width_cap", DEFAULT_AUTO_WIDTH_CAP)
            if type(floor) is not int or floor < 0:
                raise ValueError("width_floor must be a non-negative integer")
            if type(cap) is not int or cap < 0:
                raise ValueError("width_cap must be a non-negative integer")
            if floor > cap:
                raise ValueError("width_floor must not exceed width_cap")
            if (
                isinstance(target, bool)
                or not isinstance(target, int | float)
                or not math.isfinite(target)
                or target <= 0
            ):
                raise ValueError("occupancy_target must be a finite positive number")
            settings = _GateSettings(
                width=0,
                wait_seconds=float(wait_seconds),
                auto=True,
                occupancy_target=float(target),
                width_floor=floor,
                width_cap=cap,
                paused=paused,
                cut=cut,
                cut_form=cut_form,
                reason=reason,
            )
            return replace(settings, width=self._auto_width(settings))
        if type(raw_width) is not int or raw_width < 0:
            raise ValueError('width must be a non-negative integer or "auto"')
        return _GateSettings(
            raw_width,
            float(wait_seconds),
            paused=paused,
            cut=cut,
            cut_form=cut_form,
            reason=reason,
        )

    def _memory_width(self, settings: _GateSettings) -> int | None:
        """The pool arithmetic's width, clamped, or ``None`` before it can run."""
        if self._context_estimate is None or not self._pool_tokens:
            return None
        raw = math.floor(
            self._pool_tokens * settings.occupancy_target / self._context_estimate
        )
        return max(settings.width_floor, min(raw, settings.width_cap))

    def _auto_width(self, settings: _GateSettings) -> int:
        """The automatic width: the controller's choice under the pool ceiling.

        A missing reading is a state to hold across, never a zero to compute
        from: with no estimate the last width stands, and before any reading the
        floor does. The cap is the engine's own running ceiling, so the gate must
        never publish past it however large the pool arithmetic comes out.

        Until the controller has a throughput reading to reason from, the pool
        arithmetic is the width, which is what an operator has always gotten from
        ``"auto"``. Past that the pool arithmetic becomes a ceiling instead: it
        is still the figure that says how much the engine can hold, and the
        controller's measured width is clamped under it every time it is
        resolved, so a shrinking pool pulls the published width down at the same
        moment the gate would have noticed it.
        """
        memory = self._memory_width(settings)
        held = self._effective_width
        lower = min(settings.width_floor, GATE_CONTROLLER_WIDTH_FLOOR)
        if self._auto_transition_pending and self._last_integer_width is not None:
            # A live switch from an operator width to auto must not jump to the
            # memory rule. Preserve the width that was in force, then let the
            # throughput controller move it from that safe starting point.
            starting = self._last_integer_width
            width = starting if memory is None else min(starting, memory)
            self._controller_width = width
            self._controller_proposal = width
            self._controller_phase = "climb"
            self._controller_reference_width = None
            self._controller_reference_rate = None
            self._controller_best_rate = None
            self._controller_rates.clear()
            self._controller_health.clear()
        elif memory is None:
            width = held if held is not None else settings.width_floor
        elif self._controller_width is None:
            width = memory
        else:
            width = max(lower, min(self._controller_width, memory))
        self._controller_lower = lower
        self._controller_upper = memory
        self._effective_width = width
        return width

    def observe_lane(
        self,
        pool_tokens: int | None,
        mean_context: int | None,
        *,
        now: float | None = None,
        generation_tokens: int | None = None,
        running: int | None = None,
        waiting: int | None = None,
        prefix_hit_rate: float | None = None,
        kv_occupancy: float | None = None,
    ) -> None:
        """Fold one fresh lane reading into the automatic width estimate.

        The context and pool feed the pool arithmetic exactly as before; the
        counters feed the controller, which differences the cumulative
        generation counter between successive readings rather than reading any
        level out of it. A caller that supplies no counter -- a lane whose engine
        publishes none -- leaves the controller unstarted and the pool arithmetic
        in force.
        """
        stamp = time.monotonic() if now is None else now
        self._fold_capacity(pool_tokens, mean_context, stamp)
        self._step_controller(
            generation_tokens=generation_tokens,
            stamp=stamp,
            running=running,
            waiting=waiting,
            prefix_hit_rate=prefix_hit_rate,
            kv_occupancy=kv_occupancy,
        )

    def _fold_capacity(
        self, pool_tokens: int | None, mean_context: int | None, stamp: float
    ) -> None:
        """Fold one reading into the working-context estimate.

        Only a reading with traffic in it says anything about how large the
        traffic is: an idle lane reports no context, and a reading without a pool
        size reports no shape, so both are held across rather than fed in as a
        small or zero value. The estimate is a time-weighted average so a single
        unusual reading moves it by a fraction of the reading's own distance.
        """
        if pool_tokens is None or pool_tokens <= 0:
            return
        if mean_context is None or mean_context <= 0:
            return
        if self._context_estimate is None:
            self._context_estimate = float(mean_context)
        else:
            elapsed = max(0.0, stamp - self._context_stamp)
            alpha = 1.0 - math.exp(
                -elapsed / DEFAULT_AUTO_CONTEXT_TIME_CONSTANT_SECONDS
            )
            self._context_estimate += alpha * (mean_context - self._context_estimate)
        self._context_stamp = stamp
        self._pool_tokens = pool_tokens

    def _step_controller(
        self,
        *,
        generation_tokens: int | None,
        stamp: float,
        running: int | None,
        waiting: int | None,
        prefix_hit_rate: float | None,
        kv_occupancy: float | None,
    ) -> None:
        """Advance the width controller by one lane reading.

        The counter is an odometer, so the only quantity in it is the difference
        between two readings divided by the time between them. Everything below
        is that rate, judged over enough readings that its own noise is small
        against a step, with the health watches outranking it.
        """
        previous = self._throughput_tokens
        elapsed = stamp - self._throughput_stamp
        rate = (
            (generation_tokens - previous) / elapsed
            if generation_tokens is not None
            and previous is not None
            and elapsed > 0
            and generation_tokens >= previous
            else None
        )
        if generation_tokens is not None:
            self._throughput_tokens = generation_tokens
            self._throughput_stamp = stamp
        health_guard = self._record_health_sample(
            rate=rate,
            running=running,
            prefix_hit_rate=prefix_hit_rate,
            kv_occupancy=kv_occupancy,
        )
        if health_guard is not None:
            self._controller_rates.clear()
            self._step_width_down(health_guard)
            return
        if generation_tokens is None:
            return
        if previous is None or elapsed <= 0:
            self._controller_note = "memory rule: throughput needs two readings"
            return
        if generation_tokens < previous:
            # A counter below its own previous value means a new engine is
            # answering: the difference spans two processes and the interval
            # measures nothing, so the window is dropped rather than judged on a
            # rate no width produced.
            self._controller_rates.clear()
            self._controller_health.clear()
            self._controller_note = "memory rule: generation counter reset"
            return
        rate = (generation_tokens - previous) / elapsed

        if waiting is not None and waiting <= 0:
            # A width can only be scored against a lane that wants more than the
            # width allows. With nothing waiting, the reading describes the
            # demand rather than the width, and every width would look alike.
            self._controller_rates.clear()
            self._controller_note = "holding: no demand above the current width"
            return

        if self._controller_proposal is None:
            self._controller_proposal = self._current_width()
        if self._controller_proposal is None:
            self._controller_rates.clear()
            self._controller_health.clear()
            self._controller_note = "holding: no width in force to score"
            return
        self._controller_rates.append(rate)
        if len(self._controller_rates) < GATE_CONTROLLER_WINDOWS:
            if not (self._controller_note or "").startswith("stepping down:"):
                self._controller_note = (
                    f"measuring width {self._controller_proposal}: "
                    f"{len(self._controller_rates)}/{GATE_CONTROLLER_WINDOWS} windows"
                )
            return
        judged = math.fsum(self._controller_rates) / len(self._controller_rates)
        self._controller_rates.clear()
        self._judge_width(judged)

    def _record_health_sample(
        self,
        *,
        rate: float | None,
        running: int | None,
        prefix_hit_rate: float | None,
        kv_occupancy: float | None,
    ) -> str | None:
        """Judge health independently of queued demand and throughput scoring."""
        if running is None or running <= 0:
            self._controller_health.clear()
            return None
        self._controller_health.append(
            self._controller_health_sample(
                rate=rate,
                running=running,
                prefix_hit_rate=prefix_hit_rate,
                kv_occupancy=kv_occupancy,
            )
        )
        if len(self._controller_health) < GATE_CONTROLLER_WINDOWS:
            return None
        guard = self._controller_guard(self._controller_health)
        self._controller_health.clear()
        return guard

    def _judge_width(self, judged: float) -> None:
        """Choose the next width from one judged throughput reading.

        Three questions, in order: is the rise still rising, has the plateau
        started, or is the width already inside it. The objective throughout is
        the SMALLEST width that still holds the plateau, because a width above it
        costs speed on every stream and buys no aggregate throughput -- so the
        search climbs while climbing pays, and once it stops paying it walks back
        down and stops at the first width the plateau no longer survives.
        """
        current = self._controller_proposal
        upper = self._controller_upper
        if current is None:
            return
        if upper is None:
            self._controller_note = "holding: the pool ceiling is not yet known"
            return
        lower = self._controller_lower if self._controller_lower is not None else 0

        if self._controller_phase == "hold":
            self._controller_proposal = self._controller_reference_width
            best = self._controller_best_rate
            if best is None:
                return
            if judged > best + GATE_CONTROLLER_MATERIAL_TOK_S:
                # Faster at an unchanged width means the lane itself grew, so
                # there is room again and the climb re-opens from here.
                self._controller_best_rate = judged
                self._controller_phase = "climb"
            elif judged < best - GATE_CONTROLLER_MATERIAL_TOK_S:
                # Materially slower at an unchanged width: the lane shrank under
                # the width rather than the width having changed, and the safe
                # response to a degraded lane is a narrower one.
                self._controller_phase = "descend"
                self._controller_rates.clear()
                self._controller_proposal = max(lower, current - GATE_CONTROLLER_STEP)
                self._controller_width = self._controller_proposal
            return
        if self._controller_phase == "climb":
            reference = self._controller_reference_rate
            if reference is None or (
                judged > reference + GATE_CONTROLLER_MATERIAL_TOK_S
            ):
                # The rise is still paying: keep this width and reach for the
                # next one.
                self._accept(current, judged)
                if current + GATE_CONTROLLER_STEP <= upper:
                    self._propose(current + GATE_CONTROLLER_STEP, wider=True)
                else:
                    # Already at the memory arithmetic's ceiling, so the wider
                    # width the climb would ask for does not exist. What has been
                    # accepted stands, and the search turns around from it.
                    self._controller_phase = "descend"
                    self._propose(current - GATE_CONTROLLER_STEP, wider=False)
                return
            # The rise has stopped. The width in force held it, and above this
            # point every extra slot costs speed on every stream and buys no
            # aggregate throughput -- so the search turns around and walks down
            # from here.
            self._controller_phase = "descend"
            self._propose(current - GATE_CONTROLLER_STEP, wider=False)
            return

        if self._controller_phase == "descend":
            best = self._controller_best_rate
            if best is None:
                return
            if judged >= best - GATE_CONTROLLER_MATERIAL_TOK_S:
                # The narrower width still holds the plateau, and a narrower one
                # is what the objective asks for, so keep it and try the next
                # one down.
                self._accept(current, judged)
                if current - GATE_CONTROLLER_STEP >= lower:
                    self._propose(current - GATE_CONTROLLER_STEP, wider=False)
                else:
                    self._hold("holding: the narrowest width the plateau survives")
                return
            # Throughput fell materially, so the width below this one does not
            # hold the plateau. The last width that did is the answer, and this
            # reading is discarded rather than taken as its throughput.
            self._hold("holding: the narrowest width the plateau survives")
            return

    def _accept(self, width: int, rate: float) -> None:
        """Record a width as one the lane held, and the throughput it held it at."""
        self._controller_width = width
        self._controller_proposal = width
        self._controller_reference_width = width
        self._controller_reference_rate = rate
        best = self._controller_best_rate
        self._controller_best_rate = rate if best is None else max(best, rate)

    def _propose(self, width: int, *, wider: bool) -> None:
        """Put a width in force so the next window measures it."""
        lower = self._controller_lower if self._controller_lower is not None else 0
        upper = self._controller_upper
        if upper is None:
            self._controller_note = "holding: the pool ceiling is not yet known"
            return
        bounded = max(lower, min(width, upper))
        self._controller_width = bounded
        self._controller_proposal = bounded
        self._controller_note = (
            f"measuring a {'wider' if wider else 'narrower'} width {bounded}"
        )

    def _hold(self, note: str) -> None:
        """Settle on the reference width and go on checking it."""
        self._controller_phase = "hold"
        self._controller_width = self._controller_reference_width
        self._controller_proposal = self._controller_reference_width
        self._controller_note = note

    def _controller_guard(
        self,
        health: list[tuple[float | None, float | None, float | None]],
    ) -> str | None:
        """Report a health watch that fails across one controller window.

        ``None`` running readings are deliberately absent from each health
        column: SGLang publishes zero prefix hits when the lane is idle, and
        that value is not evidence of a cold cache. A loaded window must have
        the same health failure in at least half its readings before it can
        step the width down.
        """
        required = math.ceil(len(health) / 2)
        prefix = [sample[0] for sample in health if sample[0] is not None]
        if (
            sum(value < GATE_CONTROLLER_PREFIX_HIT_FLOOR for value in prefix)
            >= required
        ):
            # The prefix cache pays for the context every stream shares, and it
            # is evicted by width long before the pool fills -- measured at half
            # occupancy with preemptions still at zero -- so a sustained low hit
            # rate is the first signal that the width has gone too far.
            return (
                f"stepping down: prefix hit rate {float(min(prefix)):.3f} "
                f"is below the {GATE_CONTROLLER_PREFIX_HIT_FLOOR:.2f} floor"
            )
        kv = [sample[1] for sample in health if sample[1] is not None]
        if sum(value >= GATE_CONTROLLER_KV_CEILING for value in kv) >= required:
            # The pool ceiling is a second, blunter reading of the same
            # pressure: it fills later than the cache thrashes, which is why a
            # sustained ceiling is a guard and not the objective's measurement.
            return (
                f"stepping down: KV occupancy {float(max(kv)):.3f} is at the "
                f"{GATE_CONTROLLER_KV_CEILING:.2f} ceiling"
            )
        per_stream = [sample[2] for sample in health if sample[2] is not None]
        if (
            sum(value < GATE_CONTROLLER_PER_STREAM_FLOOR_TOK_S for value in per_stream)
            >= required
        ):
            # Aggregate throughput can hold while every stream crawls, which is
            # a lane nobody can use: the floor is on what one stream gets, and
            # one noisy scrape is not enough to declare the lane unusable.
            return (
                f"stepping down: {float(min(per_stream)):.1f} tokens/s per stream is "
                f"below the {GATE_CONTROLLER_PER_STREAM_FLOOR_TOK_S:.0f} floor"
            )
        return None

    @staticmethod
    def _controller_health_sample(
        *,
        rate: float | None,
        running: int | None,
        prefix_hit_rate: float | None,
        kv_occupancy: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Capture health values, ignoring the idle lane's sentinel readings."""
        if running is None or running <= 0:
            return (None, None, None)
        return (
            prefix_hit_rate,
            float(kv_occupancy) if kv_occupancy is not None else None,
            rate / running if rate is not None else None,
        )

    def _step_width_down(self, reason: str) -> None:
        """Step the width down on a health watch and re-plan from there."""
        current = self._current_width()
        if current is None:
            return
        lower = self._controller_lower if self._controller_lower is not None else 0
        self._controller_width = max(lower, current - GATE_CONTROLLER_STEP)
        # The watches overrule the objective's evidence rather than adding to it:
        # every judgement made at the old width measured the width that caused
        # the failure, so the search restarts from the width that survives it.
        self._controller_phase = "climb"
        self._controller_proposal = self._controller_width
        self._controller_reference_width = None
        self._controller_reference_rate = None
        self._controller_best_rate = None
        self._controller_rates.clear()
        self._controller_health.clear()
        self._controller_note = reason

    def _current_width(self) -> int | None:
        """The width in force, controller's choice first."""
        if self._controller_width is not None:
            return self._controller_width
        return self._effective_width

    async def acquire(self, receive: Receive) -> _Admission:
        """Wait in FIFO order, or report timeout/departure without a relay.

        A paused gate admits nobody: the width is ignored while the pause holds,
        so an in-flight relay is left to finish and every later caller stays in
        the FIFO. Clearing the pause releases them in arrival order through the
        same head-of-queue test, so a pause needs no separate queue.
        """
        arrived_at = time.monotonic()
        initial = self.settings()
        if initial.width == 0 and not initial.paused:
            return _Admission("bypass")

        token = object()
        deadline = asyncio.get_running_loop().time() + initial.wait_seconds
        disconnect_task = asyncio.create_task(RouterApp._wait_for_disconnect(receive))
        carry_disconnect = False
        async with self._condition:
            self._waiters.append((token, arrived_at))
            self._condition.notify_all()
        try:
            while True:
                async with self._condition:
                    if disconnect_task.done():
                        return _Admission(
                            "disconnected",
                            gate_wait_s=max(0.0, time.monotonic() - arrived_at),
                        )
                    settings = self.settings()
                    if settings.width == 0 and not settings.paused:
                        return _Admission(
                            "bypass",
                            gate_wait_s=max(0.0, time.monotonic() - arrived_at),
                        )
                    if (
                        self._waiters
                        and self._waiters[0][0] is token
                        and not settings.paused
                        and self._in_flight < settings.width
                    ):
                        self._waiters.popleft()
                        self._in_flight += 1
                        self._condition.notify_all()
                        carry_disconnect = True
                        return _Admission(
                            "acquired",
                            disconnect_task=disconnect_task,
                            gate_wait_s=max(0.0, time.monotonic() - arrived_at),
                        )

                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        return _Admission(
                            "timed-out",
                            retry_after_seconds=GENERATION_RETRY_AFTER_SECONDS,
                            gate_wait_s=max(0.0, time.monotonic() - arrived_at),
                        )
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=min(_GATE_CONFIG_REFRESH_SECONDS, remaining),
                        )
        finally:
            async with self._condition:
                with suppress(ValueError, StopIteration):
                    self._waiters.remove(
                        next(item for item in self._waiters if item[0] is token)
                    )
                self._condition.notify_all()
            if not carry_disconnect:
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)

    async def release(self) -> None:
        """Return one acquired slot and wake the FIFO head."""
        async with self._condition:
            if self._in_flight <= 0:
                raise RuntimeError("generation gate released without an acquired slot")
            self._in_flight -= 1
            self._condition.notify_all()

    async def wait_for_cut(self) -> None:
        """Return once a paused gate declares a cut, so a relay can end itself.

        The setting is re-read from the gate file rather than signalled through
        a handle, which is what lets an operator cut relays in flight, and what
        makes a cut survive a router restart: nothing has to be carried in
        memory for the next process to honour it.
        """
        while True:
            settings = self.settings()
            if settings.paused and settings.cut:
                return
            await asyncio.sleep(_CUT_POLL_SECONDS)

    def snapshot(self) -> dict[str, object]:
        settings = self.settings()
        snapshot: dict[str, object] = {
            "enabled": settings.width > 0,
            "width": settings.width,
            "effective_width": settings.width,
            "context_estimate": self._context_estimate,
            "wait_seconds": settings.wait_seconds,
            "in_flight": self.in_flight,
            "waiting": self.waiting,
            # Published so a reader sees the pause rather than inferring it from
            # a zero admitted count: a lane with nothing running and a lane that
            # is refusing to run anything look identical in the counters alone.
            "paused": settings.paused,
            "reason": settings.reason,
            "config_path": (
                str(self.config_path) if self.config_path is not None else None
            ),
        }
        if settings.auto:
            # Only an automatic width has a rule behind it to report. An
            # operator-set integer is the whole explanation of itself, so the
            # extra keys appear for the mode that chose a width rather than for
            # the mode that was handed one.
            snapshot["width_mode"] = (
                "throughput" if self._controller_width is not None else "memory"
            )
            snapshot["width_reason"] = self._controller_note
        return snapshot


def _receipt_setting(name: str, explicit: float | None, default: float) -> float:
    """Resolve one sampling setting: the argument, else the environment, else default.

    ``inf`` is accepted and means no ceiling, which is what a campaign wanting
    the full row-per-call record asks for.
    """
    raw: str | None = None
    if explicit is not None:
        value = float(explicit)
    else:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{name} must be a number, got {raw!r}") from None
    # ``not value > 0`` rather than ``value <= 0`` so a nan is refused too: every
    # comparison against nan is false, so it would otherwise reach the sink as a
    # ceiling no arrival count can stay under.
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Upstream:
    """One engine origin, optional auth header, and reported native model id."""

    base_url: str
    auth_header: tuple[str, str] | None = None
    model_id: str | None = None
    # Accelerator width, carried from the registration the serving allocation
    # wrote. Ranking reads it from here rather than from the engine's card so
    # an engine with no way to echo our own metadata back is still rankable.
    # ``None`` where the upstream was discovered without a registration.
    accelerator_count: int | None = None


class UpstreamResolver(Protocol):
    """Return the engine endpoints currently eligible for routing."""

    async def resolve(self) -> Sequence[Upstream]: ...


class DynamicUpstreamResolver:
    """Adapt a synchronous discovery supplier to the router's async interface."""

    def __init__(self, supplier: Callable[[], Sequence[Upstream]]) -> None:
        self._supplier = supplier

    async def resolve(self) -> Sequence[Upstream]:
        return await asyncio.to_thread(self._supplier)


@dataclass(frozen=True, slots=True)
class _Catalog:
    upstream: Upstream
    payload: dict[str, Any]
    cards: tuple[dict[str, Any], ...]


_Owner = tuple[Upstream, dict[str, Any]]


def _routing_rank(
    upstream: Upstream, card: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    """Score one engine by how many accelerators it holds and when it started.

    The width comes from the upstream's registration, which the serving
    allocation wrote from its own profile, and falls back to the engine's card
    where an upstream was discovered without one. Either way ``None`` records
    that the width is unknown rather than substituting a value -- an engine
    that cannot echo site metadata must not be ranked as though it were narrow.

    The start stamp is the card's ``created`` field, which an engine may omit
    or report in a form that cannot be ordered; ``None`` records that absence
    for the same reason, so a pair that only differs there stays unranked
    instead of being separated by an invented default.
    """
    accelerators = upstream.accelerator_count
    if accelerators is None:
        metadata = card.get("ambix")
        if isinstance(metadata, Mapping):
            declared = metadata.get("accelerator_count")
            accelerators = declared if type(declared) is int else None
    created = card.get("created")
    return accelerators, created if type(created) is int else None


def _reported_width(upstream: Upstream, card: Mapping[str, Any]) -> int | None:
    """Return the accelerator width for logging, or None when it is unknown.

    Shares _routing_rank's resolution so a log line never claims a width that
    ranking did not use, and never raises on an engine that carries no site
    metadata.
    """
    width, _ = _routing_rank(upstream, card)
    return width


def _preferred_owner(owners: Sequence[_Owner]) -> _Owner | None:
    """Return the engine that outranks every peer, or None when two are level.

    Several reachable engines advertising one model id is the normal state
    while a serve is being replaced by a wider one: the incoming engine joins
    the catalog the moment it answers, and both are healthy for the length of
    the overlap. Routing to the widest engine keeps that window free of a
    capacity regression, and the later start stamp breaks a tie towards the
    engine that is replacing its peer rather than the one being retired. Where
    the two leaders are indistinguishable on both terms there is no ground for
    preferring either, and the caller refuses instead of choosing arbitrarily.
    """
    ranks = [_routing_rank(upstream, card) for upstream, card in owners]
    widths = [accelerators for accelerators, _ in ranks]
    if any(width is None for width in widths):
        # One unknown width makes the whole comparison unsound: preferring a
        # known 4 over an unknown would be ranking on the absence of metadata
        # rather than on capacity. Drop the width term for everyone and let the
        # start stamp decide, which still favours the engine replacing its peer
        # and still refuses when there is no ground to choose.
        leaders = [
            (owner, started) for owner, (_, started) in zip(owners, ranks, strict=True)
        ]
    else:
        widest = max(widths)
        leaders = [
            (owner, started)
            for owner, (accelerators, started) in zip(owners, ranks, strict=True)
            if accelerators == widest
        ]
    if len(leaders) == 1:
        return leaders[0][0]
    if any(started is None for _, started in leaders):
        return None
    newest = max(started for _, started in leaders)
    latest = [owner for owner, started in leaders if started == newest]
    return latest[0] if len(latest) == 1 else None


def _by_model_id(owners: Sequence[_Owner]) -> dict[str, list[_Owner]]:
    """Partition reachable engine cards by the native model id they advertise.

    Insertion order preserves first-seen position, so a resolved duplicate
    takes the slot of its first occurrence and the collapsed union keeps a
    deterministic ordering across the source catalogs.
    """
    grouped: dict[str, list[_Owner]] = {}
    for upstream, card in owners:
        grouped.setdefault(card["id"], []).append((upstream, card))
    return grouped


class RouterApp:
    """Resolve catalog owners, then FIFO-gate generation relays.

    Catalog listing and token counting bypass decode admission. The router exposes
    no health route.
    """

    _GENERATION_PATHS = frozenset({"/v1/messages", "/v1/chat/completions"})
    _ROUTED_PATHS = frozenset(
        {
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/chat/completions",
        }
    )

    def __init__(
        self,
        resolver: UpstreamResolver,
        *,
        timeout: aiohttp.ClientTimeout | None = None,
        connection_limit: int = 2048,
        lane_document: Path | None = None,
        lane_interval: int = 30,
        request_receipts_path: Path | None = None,
        receipt_max_rows_per_s: float | None = None,
        receipt_window_s: float | None = None,
        gate_file: Path | None = None,
    ) -> None:
        self._resolver = resolver
        # Per-request attribution lands beside the lane document the launching
        # command already names, so it needs no further operator step to be
        # reachable once the router is running. Kept off entirely when there is
        # no document directory to write into, and never able to disturb
        # routing -- the sink logs a failure and discards.
        if request_receipts_path is not None:
            self._receipts_path: Path | None = request_receipts_path
        elif lane_document is not None:
            self._receipts_path = lane_document.with_name(RECEIPTS_FILENAME)
        else:
            self._receipts_path = None
        # Resolved at construction so a malformed setting stops the launch
        # rather than surfacing later as rows sampled at a rate nobody asked
        # for, and so the value in force is fixed for the process's life.
        self._receipt_max_rows_per_s = _receipt_setting(
            RECEIPT_MAX_ROWS_PER_S_ENV, receipt_max_rows_per_s, DEFAULT_MAX_ROWS_PER_S
        )
        self._receipt_window_s = _receipt_setting(
            RECEIPT_WINDOW_S_ENV, receipt_window_s, DEFAULT_WINDOW_S
        )
        self._receipts: RequestReceiptSink | None = None
        # The shared lane budget is published from here rather than from its own
        # allocation. A separate job spent a core of a 30-core GPU reservation
        # on one HTTP read every thirty seconds, while a peer's GPU work pended
        # on Resources with two cards idle -- the reservation binds on cores
        # allocated, not on cores used. This process is already standing, is
        # already blocked on IO, and already survives serve rotations, so the
        # refresh costs nothing additional. It must not live in a session:
        # a producer that dies with its coordinator stops silently and looks
        # exactly like a quiet lane.
        self._lane_document = lane_document
        self._lane_interval = lane_interval
        self._lane_task: asyncio.Task[None] | None = None
        self._generation_gate = _GenerationGate(
            gate_file
            if gate_file is not None
            else lane_document.with_name(GATE_FILENAME)
            if lane_document is not None
            else None
        )
        # Opt-in, because it logs one line per routed request. Hashes only.
        self._prefix_diagnostic = (
            os.environ.get("AMBIX_ROUTER_PREFIX_PROBE", "").strip() == "1"
        )
        self._timeout = timeout or aiohttp.ClientTimeout(total=None, connect=10)
        self._session: aiohttp.ClientSession | None = None
        # This connector ceiling protects file descriptors and deliberately
        # stays far above the generation gate. Decode admission and transport
        # capacity are separate limits; aiohttp's unset default of 100 would
        # otherwise become an invisible second queue.
        self._connection_limit = connection_limit

    async def __call__(
        self, scope: dict[str, Any], receive: Receive, send: Send
    ) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope_type != "http":
            return

        # Stamped before any branch, so a request answered here without a relay
        # still carries the time it spent in the router: a self-answered row
        # whose duration was never observed would read as instantaneous.
        began = time.perf_counter()
        method = scope.get("method")
        path = scope.get("path")
        if method == "GET" and path == "/v1/models":
            await self._serve_catalog(scope, receive, send, began)
            return
        if method != "POST" or path not in self._ROUTED_PATHS:
            await self._json_error(
                scope, receive, send, 404, "unsupported router path", began=began
            )
            return

        body, disconnected = await self._request_body(receive)
        if disconnected:
            # The caller left while still uploading, so no engine was asked
            # anything and no answer was ever sent. That is an outcome the
            # record owes rather than an early return: the request did reach
            # the router and die there, and without a row its only trace is a
            # server-side connection teardown.
            self._record_self_answer(
                scope, "", began, http_status=None, caller_gone=True
            )
            return
        try:
            payload = json.loads(body)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            await self._json_error(
                scope,
                receive,
                send,
                400,
                "request body must be valid JSON",
                began=began,
            )
            return
        model_id = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(model_id, str) or not model_id:
            await self._json_error(
                scope,
                receive,
                send,
                400,
                "request body must contain a model id",
                began=began,
            )
            return

        catalogs = await self._reachable_catalogs()
        owners: list[_Owner] = [
            (catalog.upstream, card)
            for catalog in catalogs
            for card in catalog.cards
            if card["id"] == model_id
        ]
        if not owners:
            await self._json_error(
                scope,
                receive,
                send,
                404,
                f"unknown model id: {model_id}",
                model_id=model_id,
                began=began,
            )
            return
        selected = _preferred_owner(owners)
        if selected is None:
            await self._json_error(
                scope,
                receive,
                send,
                409,
                f"duplicate model id: {model_id}",
                model_id=model_id,
                began=began,
            )
            return
        upstream, card = selected
        if len(owners) > 1:
            logger.info(
                "router preference model=%s candidates=%d origin=%s accelerators=%s",
                model_id,
                len(owners),
                upstream.base_url,
                _reported_width(upstream, card),
            )

        self._log_prefix_divergence(payload, scope)
        payload, body = self._repair_system_roles(payload, body)
        relay_body = self._clamp_output_tokens(payload, body, card)
        admission = (
            await self._generation_gate.acquire(receive)
            if path in self._GENERATION_PATHS
            else _Admission("bypass")
        )
        if admission.outcome == "disconnected":
            self._record_self_answer(
                scope,
                model_id,
                began,
                http_status=None,
                caller_gone=True,
                gate_wait_s=admission.gate_wait_s,
            )
            return
        if admission.outcome == "timed-out":
            await self._overloaded_error(
                scope,
                receive,
                send,
                model_id=model_id,
                began=began,
                retry_after_seconds=admission.retry_after_seconds,
                gate_wait_s=admission.gate_wait_s,
            )
            return

        try:
            await self._relay(
                scope,
                receive,
                send,
                relay_body,
                upstream,
                model_id=model_id,
                caller_hint=self._caller_hint(scope),
                started_at=datetime.now(UTC),
                gate_wait_s=admission.gate_wait_s,
                disconnect_task=admission.disconnect_task,
                watch_cut=admission.outcome == "acquired",
            )
        finally:
            if admission.outcome == "acquired":
                await self._generation_gate.release()

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await self._client()
                if self._lane_document is not None:
                    self._lane_task = asyncio.create_task(self._publish_lane())
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._lane_task is not None:
                    self._lane_task.cancel()
                    self._lane_task = None
                if self._session is not None:
                    await self._session.close()
                    self._session = None
                if self._receipts is not None:
                    self._receipts.close()
                    self._receipts = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _log_prefix_divergence(self, payload: object, scope: Mapping[str, Any]) -> None:
        """Record where one caller's prompt stops matching its previous turn.

        A prefix cache that is consulted and still misses is being asked about a
        prefix that genuinely differs. Measured 2026-09-15: queries ran at 1.12x
        prompt tokens, so every token was looked up, while the hit rate sat near
        37% on an almost empty lane -- which contention cannot explain. The
        remaining candidate is that something varies at the HEAD of the prompt,
        because a single changed byte early invalidates every block after it.

        Hashes only, at increasing depths, so the log carries no prompt content:
        the first depth whose digest changes between two consecutive turns is
        where reuse dies. Divergence at the shallowest depth means the very top
        of the prompt moves per request and no reuse is possible at all.
        """
        if not self._prefix_diagnostic:
            return
        try:
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if not isinstance(messages, list):
                return
            flat = json.dumps(messages, separators=(",", ":"), sort_keys=False)
        except (TypeError, ValueError):
            return
        digests = []
        for depth in (512, 2048, 8192, 32768, 131072):
            chunk = flat[:depth]
            digests.append(f"{depth}:{hashlib.sha256(chunk.encode()).hexdigest()[:8]}")
            if len(flat) <= depth:
                break
        logger.info(
            "prefix-probe consumer=%s chars=%d %s",
            self._caller_hint(scope),
            len(flat),
            " ".join(digests),
        )

    @staticmethod
    def _caller_hint(scope: Mapping[str, Any]) -> str:
        """Identify the calling process well enough to group its own turns."""
        headers = scope.get("headers") or ()
        agent = b""
        for name, value in headers:
            if bytes(name).lower() == b"user-agent":
                agent = bytes(value)
                break
        client = scope.get("client")
        host = client[0] if isinstance(client, Sequence) and client else "unknown"
        port = client[1] if isinstance(client, Sequence) and len(client) > 1 else 0
        return f"{host}:{port}|{agent.decode('utf-8', 'replace')[:32]}"

    async def _publish_lane(self) -> None:
        """Republish the shared lane budget for as long as the router runs.

        Never allowed to disturb routing: every failure publishes an explicit
        unavailable state and the loop continues, because a relay that stopped
        serving requests to keep a metrics file current would have inverted its
        own purpose.
        """
        from collections import deque

        from imas_ambix.agent.lane import (
            LaneWindow,
            detect_settling,
            fetch_lane_capacity,
            write_lane_document,
            write_unavailable_document,
        )

        # Five samples, so the published budget rests on ~2.5 minutes at the
        # default cadence rather than on one draw. The window is what makes the
        # figure a level instead of a ratio caught mid-flicker; the length is a
        # trade the reader should know about, since a genuine ramp is reported
        # late by up to that span. `settling` and `volatile` both cover that gap
        # by marking the figure an upper bound, which is the safe direction.
        readings: deque = deque(maxlen=5)
        previous = None
        while True:
            target = "unresolved"
            try:
                upstreams = await self._resolver.resolve()
                if not upstreams:
                    raise RuntimeError("no upstream is registered")
                target = f"{upstreams[0].base_url.rstrip('/')}/metrics"
                capacity = await asyncio.to_thread(
                    fetch_lane_capacity, upstreams[0].base_url
                )
            except (aiohttp.ClientError, OSError, RuntimeError, ValueError) as error:
                logger.warning(
                    "lane publisher unavailable target=%s error=%s: %s",
                    target,
                    type(error).__name__,
                    error,
                )
                write_unavailable_document(str(error), self._lane_document)
                self._publish_gate_snapshot()
                # Both histories are dropped, not just the last sample. A window
                # spanning an outage would average across a gap of unknown
                # length and publish it as a continuous measurement.
                readings.clear()
                previous = None
            except asyncio.CancelledError:
                raise
            else:
                readings.append(capacity)
                # The automatic width mode sizes the gate from the pool the
                # engine reports and the working context its sessions carry, so
                # the fresh reading is folded in here while it is known to be a
                # measurement rather than re-read from the published document
                # the gate itself is writing. The counters ride along with the
                # same reading: the decoded-token odometer is what the width
                # controller differences, the health readings are what its
                # watches trip on, and the gate's own FIFO depth is the demand
                # that says whether the width in force is the thing limiting the
                # lane or merely idle under it.
                self._generation_gate.observe_lane(
                    capacity.pool_tokens,
                    capacity.mean_context,
                    generation_tokens=capacity.generation_tokens,
                    running=capacity.running,
                    waiting=self._generation_gate.waiting,
                    prefix_hit_rate=capacity.prefix_hit_rate,
                    kv_occupancy=capacity.kv_occupancy,
                )
                write_lane_document(
                    capacity,
                    self._lane_document,
                    settling=detect_settling(previous, capacity),
                    refresh_interval=self._lane_interval,
                    window=LaneWindow(readings=tuple(readings)),
                )
                self._publish_gate_snapshot()
                previous = capacity
            await asyncio.sleep(self._lane_interval)

    def _publish_gate_snapshot(self) -> None:
        """Add router admission counts to the latest atomic lane reading."""
        if self._lane_document is None:
            return
        try:
            document = json.loads(self._lane_document.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("lane document root is not an object")
            gate_snapshot = self._generation_gate.snapshot()
            admission = self._generation_gate.admission_snapshot()
            document["router_generation_gate"] = gate_snapshot
            document["admission"] = admission
            engine_headroom = document.get("headroom")
            if isinstance(engine_headroom, int | float) and not isinstance(
                engine_headroom, bool
            ):
                document["engine_headroom"] = engine_headroom
                document["headroom"] = min(engine_headroom, admission["headroom"])
            scratch = self._lane_document.with_suffix(".gate.tmp")
            scratch.write_text(
                json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
            )
            scratch.replace(self._lane_document)
            self._lane_document.chmod(0o644)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            logger.warning(
                "generation gate lane snapshot dropped path=%s error=%s: %s",
                self._lane_document,
                type(error).__name__,
                error,
            )

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                auto_decompress=False,
                connector=aiohttp.TCPConnector(limit=self._connection_limit),
            )
        return self._session

    async def _reachable_catalogs(self) -> list[_Catalog]:
        upstreams = await self._resolver.resolve()
        results = await asyncio.gather(
            *(self._fetch_catalog(upstream) for upstream in upstreams),
            return_exceptions=True,
        )
        return [result for result in results if isinstance(result, _Catalog)]

    async def _fetch_catalog(self, upstream: Upstream) -> _Catalog:
        session = await self._client()
        headers = dict([upstream.auth_header] if upstream.auth_header else [])
        async with session.get(
            f"{upstream.base_url.rstrip('/')}/v1/models", headers=headers
        ) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"catalog returned {response.status}")
            payload = await response.json(content_type=None)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("catalog must contain a data list")

        cards: list[dict[str, Any]] = []
        for card in payload["data"]:
            if not isinstance(card, dict) or not isinstance(card.get("id"), str):
                raise ValueError("catalog cards must carry native ids")
            # Validate the site metadata when the engine carries it, and accept
            # the card when it does not. Requiring it here made routing a
            # vLLM-only surface, because the block is launch-owned data echoed
            # back through a middleware that only vLLM accepts -- so an engine
            # without that channel was unroutable regardless of health. The
            # width used for ranking now comes from the registration instead.
            if card.get("ambix") is not None:
                validate_catalog_metadata({card["id"]: card["ambix"]})
            cards.append(card)
        return _Catalog(upstream=upstream, payload=payload, cards=tuple(cards))

    async def _serve_catalog(
        self,
        scope: Mapping[str, Any],
        receive: Receive,
        send: Send,
        began: float,
    ) -> None:
        catalogs = await self._reachable_catalogs()
        if not catalogs:
            await self._json_error(
                scope,
                receive,
                send,
                503,
                "no upstream catalogs are reachable",
                began=began,
            )
            return
        owners = [
            (catalog.upstream, card) for catalog in catalogs for card in catalog.cards
        ]
        cards: list[dict[str, Any]] = []
        duplicates: list[str] = []
        for model_id, peers in _by_model_id(owners).items():
            if len(peers) == 1:
                cards.append(peers[0][1])
                continue
            preferred = _preferred_owner(peers)
            if preferred is None:
                duplicates.append(model_id)
                continue
            upstream, card = preferred
            logger.info(
                "router preference model=%s candidates=%d origin=%s accelerators=%s",
                model_id,
                len(peers),
                upstream.base_url,
                _reported_width(upstream, card),
            )
            cards.append(card)
        if duplicates:
            await self._json_error(
                scope,
                receive,
                send,
                409,
                f"duplicate model id: {', '.join(sorted(duplicates))}",
                began=began,
            )
            return
        payload = dict(catalogs[0].payload)
        payload["data"] = cards
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        caller_gone = await self._response(
            receive,
            send,
            200,
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            body,
        )
        # The listing names no model and is served from the merged catalogs, so
        # the row records what it answered without an engine behind it.
        self._record_self_answer(
            scope, "", began, http_status=200, caller_gone=caller_gone
        )

    async def _relay(
        self,
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
        body: bytes,
        upstream: Upstream,
        *,
        model_id: str,
        caller_hint: str,
        started_at: datetime,
        gate_wait_s: float = 0.0,
        disconnect_task: asyncio.Task[None] | None = None,
        watch_cut: bool = False,
    ) -> None:
        session = await self._client()
        # content-length and transfer-encoding describe the body the CLIENT
        # sent, and this relay may forward a different one: both
        # _repair_system_roles and _clamp_output_tokens re-encode the payload,
        # and re-labelling a role shortens the body by two bytes per message.
        # Forwarding the original length makes the upstream wait forever for
        # bytes that will never arrive -- the request hangs rather than failing,
        # so it reads as a slow serve rather than a malformed relay. Measured
        # 2026-09-16: every relabelled request hung for six minutes and then
        # retried, taking the lane down for agent traffic while small probes
        # that needed no rewrite passed in 0.2 s. Dropping both lets the client
        # library set the framing from the body actually being sent.
        drop = {b"host", b"content-length", b"transfer-encoding"}
        request_headers = [
            (name.decode("latin-1"), value.decode("latin-1"))
            for name, value in scope.get("headers", [])
            if name.lower() not in drop
            and (
                upstream.auth_header is None
                or name.decode("latin-1").lower() != upstream.auth_header[0].lower()
            )
        ]
        if upstream.auth_header is not None:
            request_headers.append(upstream.auth_header)
        raw_path = scope.get("raw_path", scope["path"].encode()).decode("ascii")
        query = scope.get("query_string", b"")
        target = f"{upstream.base_url.rstrip('/')}{raw_path}"
        if query:
            target = f"{target}?{query.decode('ascii')}"

        accounting = StreamAccounting()
        began = time.perf_counter()
        # Anything that is not a 2xx reads as failed, so a request the engine
        # refused is recorded as such rather than dropped or counted as a
        # success. The outcome is only rewritten by the two paths that can tell
        # better: a 2xx relay, and a caller that left mid-relay.
        status = STATUS_FAILED
        disconnected = disconnect_task or asyncio.create_task(
            self._wait_for_disconnect(receive)
        )
        # Armed only for a relay the gate admitted, so the cut reaches exactly
        # the relays holding decode width and never a catalog or token-count
        # request that consumes none.
        cut_task = (
            asyncio.create_task(self._generation_gate.wait_for_cut())
            if watch_cut
            else None
        )
        try:
            async with session.request(
                scope["method"], target, data=body, headers=request_headers
            ) as response:
                # The response's status describes what the ENGINE accepted, not
                # what the caller received. Holding it here and promoting it to
                # the row's outcome only once the body has been relayed whole is
                # what keeps the two apart: an engine that answers 200 and then
                # aborts its transport leaves this block through an exception
                # out of readany(), so an outcome taken from the headers alone
                # would record a truncated relay as a completed one.
                answered_ok = 200 <= response.status < 300
                await send(
                    {
                        "type": "http.response.start",
                        "status": response.status,
                        "headers": list(response.raw_headers),
                    }
                )
                while True:
                    next_chunk = asyncio.create_task(response.content.readany())
                    watched = {next_chunk, disconnected}
                    if cut_task is not None:
                        watched.add(cut_task)
                    done, _ = await asyncio.wait(
                        watched,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if disconnected in done:
                        status = STATUS_ABORTED
                        next_chunk.cancel()
                        await asyncio.gather(next_chunk, return_exceptions=True)
                        response.close()
                        return
                    if cut_task is not None and cut_task in done:
                        # A cut is not a completed turn: the relay stops reading
                        # the upstream, cancels it, and answers the client in the
                        # form the operator chose. The receipt records ABORTED so
                        # a cut turn is never recorded as one that finished.
                        status = STATUS_ABORTED
                        next_chunk.cancel()
                        await asyncio.gather(next_chunk, return_exceptions=True)
                        response.close()
                        if self._generation_gate.settings().cut_form == "error":
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": _cut_error_frame(),
                                    "more_body": True,
                                }
                            )
                            await send({"type": "http.response.body", "body": b""})
                        # The close form deliberately sends no terminal chunk: an
                        # ASGI app that returns without completing its response
                        # makes the server drop the connection, and a streaming
                        # client retries a stream closed mid-stream, keeping the
                        # turn alive rather than ending it.
                        return
                    chunk = next_chunk.result()
                    if not chunk:
                        break
                    accounting.feed(chunk)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": True,
                        }
                    )
                accounting.finish()
                if answered_ok:
                    status = STATUS_COMPLETED
                await send({"type": "http.response.body", "body": b""})
        except asyncio.CancelledError:
            # The router is shutting down under an in-flight request; the
            # caller's answer is incomplete, which is exactly what the row
            # should say.
            status = STATUS_ABORTED
            raise
        finally:
            disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)
            if cut_task is not None:
                cut_task.cancel()
                await asyncio.gather(cut_task, return_exceptions=True)
            self._record_receipt(
                accounting=accounting,
                status=status,
                model_id=model_id,
                upstream=upstream.base_url,
                caller_hint=caller_hint,
                started_at=started_at,
                began=began,
                gate_wait_s=gate_wait_s,
            )

    def _receipt_sink(self) -> RequestReceiptSink | None:
        """The process's receipt sink, built on first use.

        Built lazily so a router with no receipts path pays nothing, and
        constructed without IO so a bad path cannot fail a request.
        """
        if self._receipts_path is None:
            return None
        if self._receipts is None:
            self._receipts = RequestReceiptSink(
                self._receipts_path,
                max_rows_per_s=self._receipt_max_rows_per_s,
                window_s=self._receipt_window_s,
            )
            # Publish the ceiling in force. Sampling is visible in the rows only
            # to a reader who already suspects it, and a bound inferable from
            # behaviour alone is one that a later reader measures from scratch.
            logger.info(
                "request receipts sink path=%s max_rows_per_s=%s window_s=%s",
                self._receipts.path,
                self._receipt_max_rows_per_s,
                self._receipt_window_s,
            )
        return self._receipts

    def _record_receipt(
        self,
        *,
        accounting: StreamAccounting,
        status: str,
        model_id: str,
        upstream: str,
        caller_hint: str,
        started_at: datetime,
        began: float,
        gate_wait_s: float = 0.0,
    ) -> None:
        """Append the row for one request, whatever its outcome and whoever answered it.

        Never raises into the relay: this runs in a ``finally`` whose exception
        may still be propagating, so a failure here would replace the real
        error with a bookkeeping one.
        """
        sink = self._receipt_sink()
        if sink is None:
            return
        try:
            sink.record(
                model=model_id,
                upstream=upstream,
                caller_hint=caller_hint,
                status=status,
                duration_s=time.perf_counter() - began,
                accounting=accounting,
                gate_wait_s=gate_wait_s,
                timestamp=started_at,
            )
        except (OSError, TypeError, ValueError) as error:
            logger.warning(
                "request receipt dropped model=%s error=%s: %s",
                model_id,
                type(error).__name__,
                error,
            )

    def _record_self_answer(
        self,
        scope: Mapping[str, Any],
        model_id: str,
        began: float,
        *,
        http_status: int | None,
        caller_gone: bool,
        gate_wait_s: float = 0.0,
    ) -> None:
        """Record a request this process answered without relaying it.

        A request refused before any engine was chosen -- an unroutable path, a
        body that is not JSON, a missing, unknown or ambiguous model id -- and
        the catalog listing are answered here, so no engine served them and
        there is no usage to report. So is a caller that left while still
        uploading, for which no answer was ever sent. They belong in the record
        anyway: a record of what the router served that holds only what it did
        not forward omits precisely the requests a caller reports as broken,
        whose only other trace is a log line. The row carries the model id the
        caller named, and the empty string where none was ever read.

        The upstream is the self-answered sentinel rather than an engine origin,
        so a reader attributing rows per upstream never folds the router's own
        answers into a sink's traffic.

        The status is what the caller was sent, not what the router composed:
        a departure reported before the answer was handed over records as
        ``aborted`` whatever status the router put on it. That is a statement
        about the send and not about receipt, and ``_response`` states exactly
        what a ``completed`` row does and does not promise.

        ``http_status`` is the status the router composed, or None when the
        caller had gone before any answer was sent. ``caller_gone`` is the
        departure the answer's own send observed -- the caller's side of the
        exchange, which the router's own status line does not describe: a
        refusal answered to a caller that is no longer there is not the same
        event as a refusal received, and recording both as ``failed`` makes the
        record unable to distinguish a caller that got an empty answer from one
        that got nothing.
        """
        if caller_gone or http_status is None:
            status = STATUS_ABORTED
        elif 200 <= http_status < 300:
            status = STATUS_COMPLETED
        else:
            status = STATUS_FAILED
        self._record_receipt(
            accounting=StreamAccounting(),
            status=status,
            model_id=model_id,
            upstream=SELF_ANSWERED_UPSTREAM,
            caller_hint=self._caller_hint(scope),
            started_at=datetime.now(UTC),
            began=began,
            gate_wait_s=gate_wait_s,
        )

    @staticmethod
    def _repair_system_roles(
        payload: Mapping[str, Any], body: bytes
    ) -> tuple[Mapping[str, Any], bytes]:
        """Re-label mid-conversation system messages so the prefix can be reused.

        The DeepSeek-V4.1 encoder re-emits the ENTIRE tool block after every
        message whose role is ``system``, and agent harnesses inject one such
        message per turn -- a session hook, server instructions, a remaining-token
        notice. With a large tool set that rewrites tens of thousands of tokens at
        a fresh position on every request, so no two consecutive turns share a
        prefix and the cache can never be consulted usefully.

        Measured on this deployment 2026-09-16, one worker, identical task:
        conversational turns reused 1.6-4.2% of their prompt with the roles as
        sent, and 99.3-99.7% with them re-labelled. An ablation isolated the
        cause: removing ``context_management``, ``output_config`` or ``metadata``
        changed nothing, while re-labelling alone moved a 48,159-token turn from
        0.0% to 99.4%. Across a fleet the waste was the dominant cost of serving
        -- a width-1 phase spent 23.0 of 23.7 minutes of card time re-computing a
        prefix the engine would otherwise have had.

        ``user`` is the encoder's own reading rather than an invention: it treats
        a mid-conversation system message as a user turn when deciding where the
        assistant header goes, and says so. Only messages after the first are
        touched, because a leading system message is the conventional way to open
        a conversation and the encoder handles it without re-emitting tools. The
        body is re-encoded only when a message actually changed, so an untouched
        request passes through byte-for-byte.
        """
        messages = payload.get("messages") if isinstance(payload, Mapping) else None
        if not isinstance(messages, list):
            return payload, body
        repaired: list[Any] = []
        changed = False
        for index, message in enumerate(messages):
            if (
                index > 0
                and isinstance(message, Mapping)
                and message.get("role") == "system"
            ):
                amended = dict(message)
                amended["role"] = "user"
                repaired.append(amended)
                changed = True
            else:
                repaired.append(message)
        if not changed:
            return payload, body
        amended_payload = dict(payload)
        amended_payload["messages"] = repaired
        logger.info(
            "router prefix-repair relabelled=%d messages=%d",
            sum(
                1
                for index, message in enumerate(messages)
                if index > 0
                and isinstance(message, Mapping)
                and message.get("role") == "system"
            ),
            len(messages),
        )
        return amended_payload, json.dumps(
            amended_payload, ensure_ascii=False, separators=(",", ":")
        ).encode()

    @staticmethod
    def _clamp_output_tokens(
        payload: Mapping[str, Any], body: bytes, card: Mapping[str, Any]
    ) -> bytes:
        """Cap requested output so a prompt plus response fits the model window.

        The engine rejects a request whose declared output exceeds what the
        model window leaves after the prompt. Agent harnesses fix a large
        output reservation at launch and do not re-derive it when the model is
        switched mid-session, so a prompt that fits one engine can overflow a
        narrower one merely because the old reservation was carried over.
        Leaving a quarter of the window for the response mirrors the launcher
        convention and bounds every request without tokenizing the prompt here;
        a prompt beyond three quarters of the window is handled by the engine
        as before. The body is re-encoded only when a cap actually applied, so
        otherwise the request passes through byte-for-byte.
        """
        maximum = card.get("max_model_len")
        if not isinstance(maximum, int) or maximum <= 0:
            return body
        ceiling = max(1, maximum // 4)
        clamped: dict[str, Any] | None = None
        for field in ("max_tokens", "max_completion_tokens"):
            requested = payload.get(field)
            if isinstance(requested, int) and requested > ceiling:
                if clamped is None:
                    clamped = dict(payload)
                clamped[field] = ceiling
        if clamped is None:
            return body
        return json.dumps(clamped, ensure_ascii=False, separators=(",", ":")).encode()

    @staticmethod
    async def _request_body(receive: Receive) -> tuple[bytes, bool]:
        parts: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return b"", True
            if message["type"] != "http.request":
                continue
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                return b"".join(parts), False

    @staticmethod
    async def _wait_for_disconnect(receive: Receive) -> None:
        while True:
            if (await receive())["type"] == "http.disconnect":
                return

    @staticmethod
    async def _response(
        receive: Receive,
        send: Send,
        status: int,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
    ) -> bool:
        """Send an answer this process composed, and report if the caller had gone.

        The departure is read on the same channel the relay reads, and read
        while the answer is still unsent: uvicorn marks a response complete
        inside the send that writes its body, and its channel then answers
        http.disconnect for a completed response exactly as it does for a
        caller that left, so a watch read after the hand-over reports every
        caller as gone. The hand-over carries no signal of its own either: a
        server that finds the caller gone when it writes the body returns from
        that send without raising, so the outcome of the send reports nothing
        either way.

        The sample is therefore taken here, before the answer changes hands,
        and the row's ``completed`` is what the answer's own send can promise
        -- the router composed the answer and handed it over with no departure
        reported first. It is not a promise that the caller read it, which
        nothing on this side of the server can state, and a departure landing
        in the gap between this read and the hand-over is not observable. A
        reader summing completed rows is counting answers sent, not answers
        received.

        Returns True when the caller had already gone, so the row records what
        the caller's side of the exchange showed rather than the status the
        router composed.
        """
        watcher = asyncio.create_task(RouterApp._wait_for_disconnect(receive))
        try:
            await send(
                {"type": "http.response.start", "status": status, "headers": headers}
            )
            # One turn of the loop, so a watcher that has an answer to give --
            # the caller is already gone -- gives it before the body is sent.
            # This is the last read the channel can answer: once the response
            # is complete it reports a departure for every caller alike.
            await asyncio.sleep(0)
            caller_gone = watcher.done()
            await send({"type": "http.response.body", "body": body})
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        return caller_gone

    async def _json_error(
        self,
        scope: Mapping[str, Any],
        receive: Receive,
        send: Send,
        status: int,
        detail: str,
        *,
        model_id: str = "",
        began: float,
    ) -> None:
        """Answer a request this process refuses, and record that it did.

        Every error response the router produces itself leaves through here, so
        a new refusal path cannot widen what it answers without widening the
        record too -- which a call site remembered per branch cannot promise.
        """
        body = json.dumps(
            {"error": {"message": detail}}, separators=(",", ":")
        ).encode()
        caller_gone = await self._response(
            receive,
            send,
            status,
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            body,
        )
        self._record_self_answer(
            scope, model_id, began, http_status=status, caller_gone=caller_gone
        )

    async def _overloaded_error(
        self,
        scope: Mapping[str, Any],
        receive: Receive,
        send: Send,
        *,
        model_id: str,
        began: float,
        retry_after_seconds: int,
        gate_wait_s: float = 0.0,
    ) -> None:
        """Return the native overload shape after a generation wait expires."""
        body = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "router generation queue wait limit exceeded",
                },
            },
            separators=(",", ":"),
        ).encode()
        caller_gone = await self._response(
            receive,
            send,
            529,
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(retry_after_seconds).encode()),
            ],
            body,
        )
        self._record_self_answer(
            scope,
            model_id,
            began,
            http_status=529,
            caller_gone=caller_gone,
            gate_wait_s=gate_wait_s,
        )


def create_router_app(
    resolver: UpstreamResolver,
    *,
    lane_document: Path | None = None,
    gate_file: Path | None = None,
    request_receipts_path: Path | None = None,
    receipt_max_rows_per_s: float | None = None,
    receipt_window_s: float | None = None,
) -> RouterApp:
    """Build the ASGI application around an injected upstream resolver."""
    return RouterApp(
        resolver,
        lane_document=lane_document,
        gate_file=gate_file,
        request_receipts_path=request_receipts_path,
        receipt_max_rows_per_s=receipt_max_rows_per_s,
        receipt_window_s=receipt_window_s,
    )


def serve_router(
    resolver: UpstreamResolver,
    *,
    host: str = "0.0.0.0",
    port: int,
    lane_document: Path | None = None,
    gate_file: Path | None = None,
    request_receipts_path: Path | None = None,
    receipt_max_rows_per_s: float | None = None,
    receipt_window_s: float | None = None,
) -> None:
    """Run the router ASGI application with the serving runtime."""
    import uvicorn

    # uvicorn configures its own loggers and leaves everyone else on the root
    # logger, which defaults to WARNING -- so every INFO this module emits was
    # dropped before reaching the job log. Measured 2026-09-15: 32 requests
    # routed with zero lines from this module, including the upstream-preference
    # diagnostic that has been silently absent since it was written. A
    # diagnostic that cannot be read is not a diagnostic, so attach a handler
    # rather than assume one.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    uvicorn.run(
        create_router_app(
            resolver,
            lane_document=lane_document,
            gate_file=gate_file,
            request_receipts_path=request_receipts_path,
            receipt_max_rows_per_s=receipt_max_rows_per_s,
            receipt_window_s=receipt_window_s,
        ),
        host=host,
        port=port,
    )
