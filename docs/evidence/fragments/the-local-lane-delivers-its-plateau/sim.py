"""Closed simulation of the local lane: workers, the router gate, and an admission rule.

Each worker alternates between thinking (tools) and one request. A request waits at
the gate while `width` requests are in service, then generates its tokens at the
per-stream rate measured for the number in service. A worker ends after its sampled
number of requests; the admission rule decides how many replacements launch.

usage: sim.py <calibration.json> [--hours H] [--seed S] [--width W]
Think times, work per request, requests per run and the rate curve come from
calibrate.py, so the simulation uses the lane's own measured behaviour.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random

DT = 0.25
PUBLISH_EVERY = 30.0  # the lane document's period
ADMIT_EVERY = 30.0  # the admission service's pass period
LAUNCH_POOL = 8  # launches one admission pass can start
WINDOW = 900.0
HORIZON = 180.0


class Lane:
    def __init__(self, cal, width, rule, seed, hours, fixed_n=None, queue=10**9):
        self.rng = random.Random(seed)
        self.cal = cal
        self.width = width
        self.rule = rule
        self.fixed_n = fixed_n
        self.queue = queue
        self.t = 0.0
        self.end = hours * 3600.0
        self.workers = {}
        self.next_id = 0
        self.gate = collections.deque()
        self.in_service = set()
        self.launching = []  # times at which a launched worker makes its first request
        self.intervals = []  # (run, start, end or None) service intervals, for the ledger
        self.first_seen = {}
        self.last_seen = {}
        self.slots = 0
        self.stats = collections.Counter()
        self.waits = []
        self.level_time = collections.Counter()
        self.live_time = 0.0
        curve = {int(k): v for k, v in cal["rate"].items()}
        top = max(curve)
        self.curve = lambda k: curve.get(k) or (curve[top] * top / k if k > top else curve[min(curve)])

    # -- workers ---------------------------------------------------------
    def launch(self):
        self.launching.append(self.t + self.rng.uniform(45.0, 120.0))

    def _start_worker(self):
        wid = self.next_id
        self.next_id += 1
        self.workers[wid] = {"left": self.rng.choice(self.cal["per_run"]), "state": "think", "until": self.t}

    def _new_request(self, wid):
        tokens, ttft = self.rng.choice(self.cal["work"])
        self.workers[wid].update(state="wait", tokens=float(max(tokens, 1)), ttft=ttft, arrived=self.t)
        self.gate.append(wid)
        self.first_seen.setdefault(wid, self.t)
        self.last_seen[wid] = self.t

    # -- the ledger the router keeps ---------------------------------------
    def _ledger(self):
        lo = self.t - WINDOW
        self.intervals = [iv for iv in self.intervals if iv[2] is None or iv[2] > lo]
        busy = collections.Counter()
        for run, start, end in self.intervals:
            busy[run] += (self.t if end is None else end) - max(start, lo)
        observed = min(WINDOW, self.t) or 1.0
        window_runs = len(busy)
        live_now = [
            w
            for w in self.workers
            if self.workers[w]["state"] in ("wait", "serve") or self.t - self.last_seen.get(w, -1e9) <= HORIZON
        ]
        live_span = sum(self.t - max(self.first_seen[w], lo) for w in live_now if w in self.first_seen)
        ended_span = sum(
            self.last_seen[w] - max(self.first_seen[w], lo)
            for w in busy
            if w not in self.workers and self.last_seen[w] > lo
        )
        total_busy = sum(busy.values())
        return dict(
            busy=total_busy,
            observed=observed,
            window_runs=window_runs,
            live_now=len(live_now),
            span=max(1.0, live_span + ended_span),
            mean_in_service=total_busy / observed,
        )

    def publish(self):
        led = self._ledger()
        c = self.width
        if self.fixed_n is not None:
            self.slots = max(0, self.fixed_n - len(self.workers) - len(self.launching))
            return
        if led["window_runs"] < 3 or self.t < 300 or led["busy"] <= 0:
            self.slots = max(0, c - led["live_now"] - len(self.launching))
            return
        kind, kappa = self.rule
        if kind == "window":  # the rule as it stands: every run seen in the window counts as live
            ratio = round(led["busy"] / led["observed"] / led["window_runs"], 3) or 1.0
            self.slots = max(0, math.floor(c / ratio) - led["window_runs"] - len(self.launching))
        elif kind == "live":  # duty over each run's own live span, against the runs live now
            duty = led["busy"] / led["span"]
            self.slots = max(0, math.floor(kappa * c / duty) - led["live_now"] - len(self.launching))
        elif kind == "gate":  # feed the gate until it is rarely below its width
            below = self._recent_below()
            step = 0 if below < kappa else max(1, round(below * led["live_now"]))
            self.slots = max(0, step - len(self.launching))

    def _recent_below(self):
        recent = self.recent[-int(120 / DT):]
        return sum(1 for level in recent if level < self.width) / max(1, len(recent))

    # -- one run ----------------------------------------------------------------
    def run(self, start_workers):
        self.recent = []
        for _ in range(start_workers):
            self._start_worker()
        next_publish = next_admit = 0.0
        tokens_out = 0.0
        warm = 1800.0
        while self.t < self.end:
            if self.t >= next_publish:
                self.publish()
                next_publish += PUBLISH_EVERY
            if self.t >= next_admit:
                for _ in range(min(self.slots, LAUNCH_POOL, self.queue)):
                    self.launch()
                    self.queue -= 1
                    self.slots -= 1
                next_admit += ADMIT_EVERY
            due = [x for x in self.launching if x <= self.t]
            self.launching = [x for x in self.launching if x > self.t]
            for _ in due:
                self._start_worker()
            for wid, w in list(self.workers.items()):
                if w["state"] == "think" and w["until"] <= self.t:
                    if w["left"] <= 0:
                        del self.workers[wid]
                        continue
                    w["left"] -= 1
                    self._new_request(wid)
            while self.gate and len(self.in_service) < self.width:
                wid = self.gate.popleft()
                w = self.workers[wid]
                w["state"] = "serve"
                self.in_service.add(wid)
                self.intervals.append([wid, self.t, None])
                if self.t > warm:
                    self.waits.append(self.t - w["arrived"])
            k = len(self.in_service)
            rate = self.curve(k) if k else 0.0
            for wid in list(self.in_service):
                w = self.workers[wid]
                if w["ttft"] > 0:
                    w["ttft"] -= DT
                    continue
                w["tokens"] -= rate * DT
                if self.t > warm:
                    tokens_out += rate * DT
                if w["tokens"] <= 0:
                    self.in_service.discard(wid)
                    for iv in reversed(self.intervals):
                        if iv[0] == wid and iv[2] is None:
                            iv[2] = self.t
                            break
                    self.last_seen[wid] = self.t
                    w.update(state="think", until=self.t + self.rng.choice(self.cal["think"]))
            self.recent.append(k)
            if len(self.recent) > 4000:
                del self.recent[:2000]
            if self.t > warm:
                self.level_time[k] += DT
                self.live_time += len(self.workers) * DT
            self.t += DT
        span = self.end - warm
        waits = sorted(self.waits) or [0.0]
        q = lambda p: waits[int(p * (len(waits) - 1))]
        mean = sum(k * v for k, v in self.level_time.items()) / span
        return dict(
            tok_s=tokens_out / span,
            mean_in_service=mean,
            below_9=sum(v for k, v in self.level_time.items() if k < 9) / span,
            at_width=self.level_time[self.width] / span,
            wait_p50=q(0.5),
            wait_p90=q(0.9),
            wait_p99=q(0.99),
            workers=self.live_time / span,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--out")
    args = ap.parse_args()
    cal = json.load(open(args.calibration))
    cases = [(f"fixed {n} workers", ("fixed", n)) for n in (12, 16, 20, 24, 28, 32, 36, 44)]
    cases += [
        ("rule as it stands", ("window", 1.0)),
        ("live runs, duty per live span", ("live", 1.0)),
        ("live runs, capacity x1.1", ("live", 1.1)),
        ("live runs, capacity x1.25", ("live", 1.25)),
        ("live runs, capacity x1.5", ("live", 1.5)),
        ("feed while below width >10% of 2 min", ("gate", 0.10)),
        ("feed while below width >25% of 2 min", ("gate", 0.25)),
    ]
    rows = []
    head = f"{'case':40s} {'tok/s':>6s} {'in svc':>6s} {'<9':>6s} {'at W':>6s} {'wait p50':>8s} {'p90':>6s} {'p99':>6s} {'workers':>7s}"
    print(f"width {args.width}, {args.hours:g} h after a 30 min warm-up, mean of {args.seeds} seeds")
    print(head)
    for name, rule in cases:
        acc = collections.Counter()
        for seed in range(args.seeds):
            fixed = rule[1] if rule[0] == "fixed" else None
            lane = Lane(cal, args.width, rule, seed, args.hours + 0.5, fixed_n=fixed)
            for key, value in lane.run(fixed or args.width).items():
                acc[key] += value / args.seeds
        rows.append(dict(case=name, **acc))
        print(
            f"{name:40s} {acc['tok_s']:6.0f} {acc['mean_in_service']:6.2f} {acc['below_9']:6.1%} {acc['at_width']:6.1%}"
            f" {acc['wait_p50']:8.1f} {acc['wait_p90']:6.1f} {acc['wait_p99']:6.1f} {acc['workers']:7.1f}"
        )
    if args.out:
        json.dump(dict(width=args.width, hours=args.hours, seeds=args.seeds, rows=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
