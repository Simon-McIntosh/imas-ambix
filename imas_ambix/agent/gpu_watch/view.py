"""gpu-watch — windowed, colour GPU+host load on 98dci4-gpu-0003.

A single nvidia-smi snapshot lies about a bursty encode (it swings 0↔100%
each second), so this samples over a window (default 30 s) on the node and
renders instantaneous + window-mean util%/mem% as colour bars. The mean
columns carry the overbar (x̄).

In --live mode the screen refreshes every second; a single long-lived on-node
sampler streams one reading per second, and the mean columns are a ROLLING mean
over the last *window* seconds (default 30) — so you see real-time bursts in the
"now" columns and smoothed load in the mean columns at the same time. The
sampler is ONE SLURM step for the whole session (not one per second), so it can
never exhaust a job's lifetime step counter.

The node's cards are split between two reservation groups. SLURM cgroups hide
the other group's devices from a job, so only GPUs in the running step's
cgroup are sampled — the other group's GPU util is invisible to us. ALL
jobs on the node are shown in the job table via squeue (any user, any
partition, any reservation); Group B rows (account grpb) are highlighted
in cyan so load can be inferred from their gres allocation and elapsed
time, even though per-GPU util% is not available.

The serve panel between the card rows and the job table is rendered from the
recorder's own published sources -- `imas-ambix agent watch --json --live`,
which copies the router's lane document and differences the recorder's newest
raw receipt file -- and never from a scrape of the engine. A scrape sees only
the requests that reached the engine, so it cannot show the requests the
generation gate is holding or the headroom orchestrators dispatch against, and
it stops answering at all once the engine binds loopback. The period ledger
under the serve grid is a different read -- the document the fleet publishes
every 300 s -- and this process computes none of it.

The callers table under the serve grid says whose requests fill the gate. Its
figures are the live document's `callers` block, built from the router's
per-request record; the only thing this process adds is a name for each row,
looked up off the render thread: the project a coordinator session's transcript
lives under in this account's ~/.claude/projects, or a reckon session named
after one of this account's reckon projects. Either makes the row this user's.
A row with no session is named by its host, and its user column lists the
other accounts holding jobs on that node -- the router sees an address, never
an account, so a row whose owner cannot be shown is marked `?`.

Layout. The queue is what a reader scrolls for, so the panels above it are
kept to what changes: one row per readable card, one row for the cards that
cannot be read, a single line when no caller finished a request, and a ledger
coverage column only when a period is short of record. In --live mode the job
table takes every line the screen has left, and when that would be fewer than
ten rows the serve grid closes the blank lines between its bands.

Usage:
    gpu-watch                 # one-shot 30 s-averaged reading
    gpu-watch 5               # one-shot 5 s window
    gpu-watch --live          # 1 s refresh, 30 s rolling mean (the live view)
    gpu-watch 10 --live       # 1 s refresh, 10 s rolling mean
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import os
import re
import subprocess
import sys
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

NODE = "98dci4-gpu-0003"
N_GPUS = 8  # physical H200 cards on NODE — the panel shows a row per physical card
# vLLM serving port on the node (api-base in pyproject [sn-compose]). Its
# Prometheus /metrics endpoint is the one live-load signal reachable when
# both srun --overlap (step limit) and direct ssh (kex refused) are dead.
VLLM_PORT = 18800
# realpath, not abspath: the command is reached through a symlink on PATH,
# and the sampler is a sibling of the REAL file, not of the link.
SAMPLER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sampler.py")
USER = getpass.getuser()

# The node's 8 GPUs are split into two reservation groups. SLURM cgroups hide
# the other group's devices, so per-GPU util is only ever sampleable for the
# group our own running job reserved; the other group's load is inferred from
# the node-wide job table. Index 0-3 → Group A, 4-7 → Group B.
GROUP_RESV = {"gpu_0003_grpa": "A", "gpu_0003_grpb": "B"}
GROUP_ACCT = {"grpa": "A", "grpb": "B"}


def _resv_group(reservation: str) -> str | None:
    return GROUP_RESV.get((reservation or "").strip().lower())


def _acct_group(account: str) -> str | None:
    return GROUP_ACCT.get((account or "").strip().lower())


def _run(cmd: list[str], timeout: int = 120) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout
    except Exception:
        return ""


def vllm_load() -> dict | None:
    """Read vLLM Prometheus /metrics — the reachable live-load proxy.

    Returns gauges that reflect how busy the served model is (its GPUs are
    Group A): KV-cache occupancy, in-flight + queued requests, and cumulative
    generation tokens (a rate can be derived across live ticks). Returns
    ``None`` if the endpoint is unreachable. No per-GPU util — this is the
    fallback for when the cgroup sampler can't create a step.
    """
    import urllib.request

    # The engine-specific parsing below only ever knew `vllm:` names, so this
    # returned None for every SGLang deployment -- a silently dead fallback.
    # _scrape_serve() knows both families and resolves the port from the
    # serve's own registration rather than a constant that retires with a model.
    shared = _scrape_serve()
    if shared is not None:
        pool = shared.get("pool")
        return {
            "kv": pool if pool is not None and pool <= 1.0 else (pool or 0.0) / 100.0,
            "running": shared.get("running", 0.0),
            "waiting": shared.get("queued", 0.0),
            "gen_tokens": shared.get("decode_steps", 0.0),
        }

    url = f"http://{NODE}:{VLLM_PORT}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None

    wanted = {
        "vllm:kv_cache_usage_perc": "kv",
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:generation_tokens_total": "gen_tokens",
    }
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = wanted.get(name)
        if key is None or key in out:
            continue
        try:
            out[key] = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
    return out or None


# ---------------------------------------------------------------- serve panel
# The served model's own health, between the card panel and the job table.
#
# Why this panel exists separately from the card rows: 100% GPU utilisation
# tells you the cards are busy, not that the lane is healthy. A serve whose
# prefix cache has stopped matching runs at 100% util while every worker
# starves -- measured on this node 2026-09-16, where a client-side defect drove
# reuse to 0%, aggregate output to 32.9 tok/s and time-to-first-token to 10 s
# with the cards pinned the whole time. The three numbers that would have shown
# it at a glance are here: median per-worker rate, cache hit, and TTFT.

# Candidate serve ports, newest first. The port moves when the deployment
# rotates, and a single hard-coded one reads as "serve down" forever after. The
# endpoint document is consulted first because the serve itself registers there.
ENDPOINT_DOC = "/home/ITER/mcintos/public/imas-ambix/endpoints.json"
SERVE_PORTS = (18810, 18801, 18800)
SERVE_REFRESH = 5.0  # seconds between live-document reads
MODEL_REFRESH = 60.0  # seconds between /v1/models reads
# The serve panel is rendered from the recorder's published live document
# rather than from a scrape of the engine. The command is not on PATH under a
# non-interactive shell, so the environment names it and the main checkout's
# console script is the fallback; a bare PATH lookup would read as a dead panel.
IMAS_AMBIX_CLI = os.environ.get(
    "IMAS_AMBIX_CLI", "/home/ITER/mcintos/Code/imas-ambix/.venv/bin/imas-ambix"
)
LIVE_TIMEOUT = 20.0  # seconds bounding one `agent watch --json --live`
# Rates and medians are differences across a window. At 5 s that window often
# contains no COMPLETED request -- these turns run 20-40 s -- so the cache and
# TTFT cells blanked and reappeared tick to tick. Deriving them across a
# 45 s trailing window instead means every tick has completions to divide by,
# and the numbers hold still long enough to read.
SERVE_WINDOW = 45.0

# Per-worker decode reference, tok/s: the best single-stream rate measured on
# this deployment. The median-per-worker colour ramp is scaled to it, so the
# figure answers "is each worker getting a fair share of what one worker can
# have" rather than an absolute that means nothing without the model.
PER_WORKER_REF = 40.0

# Die temperature at which these cards begin to clock down. It sets the
# temperature ramp and is printed on that column's label, because a reading
# coloured against an unstated ceiling cannot be judged.
THROTTLE_C = 84.0

_serve_cache: dict = {
    "at": 0.0,
    "raw": None,
    "model_at": 0.0,
    "model": None,
    "history": [],
    "rates": None,
    "cards": None,
}

# ----------------------------------------------------------- period ledger
# The rate rows answer "is the lane healthy now". They cannot answer "what has
# this lane actually carried", which is what sizing the deployment against a
# metered provider needs. That integral is no longer computed here: the fleet
# publishes it, every 300 s, at WATCH_DOC, derived in the repository command
# from the recorder's own receipts. This panel renders the published figures and
# nothing else, so the two tools can never disagree about the same period.
#
# The document is read fresh each render. A document older than
# LEDGER_STALE_AFTER is shown beside its own age rather than withheld -- a
# figure that is an hour old is still a reading of the lane, but a reader has to
# see the age before trusting it. A missing or unparsable document is stated in
# the panel rather than rendered as an empty ledger.
WATCH_DOC = "/home/ITER/mcintos/public/imas-ambix/watch.json"
LEDGER_STALE_AFTER = 600.0  # twice the publisher's 300 s cadence


def fmt_tokens(n: float) -> str:
    """Token count at three significant figures on the SI ladder, k to T.

    The prefix steps up as soon as the value would round to 1 in the next one,
    which is what keeps three significant figures from printing an exponent:
    999.6M formatted in M is ``1e+03``, and the same value in G is ``1``.
    """
    n = float(n or 0.0)
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        v = n / scale
        if abs(v) >= 0.9995:
            return f"{v:.3g}{suffix}"
    return f"{n:.0f}"


def fmt_usd(v: float) -> str:
    """Dollars: cents below $1000, whole dollars above, always rounded up at .5."""
    v = float(v or 0.0)
    if abs(v) < 1000:
        return f"${v:.2f}"
    return f"${int(v + 0.5)}"


def fmt_span(seconds: float) -> str:
    """Coarse duration — what a ledger row is actually backed by."""
    s = max(0.0, float(seconds or 0.0))
    for scale, suffix in ((86400.0, "d"), (3600.0, "h"), (60.0, "m")):
        if s >= scale:
            return f"{int(s // scale)}{suffix}"
    return f"{int(s)}s"


def _norm_model(name: str) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def _serve_origin() -> tuple[str, int] | None:
    """Resolve host and port from the serve's own registration, else guess."""
    try:
        import json

        with open(ENDPOINT_DOC, encoding="utf-8") as fh:
            doc = json.load(fh)
        lanes = {
            o.get("port")
            for o in doc.get("routing_origins", [])
            if isinstance(o.get("port"), int)
        }
        for entry in doc.get("endpoints", []):
            port = entry.get("port")
            host = entry.get("host") or NODE
            if isinstance(port, int):
                count = entry.get("accelerator_count")
                _serve_cache["cards"] = count if isinstance(count, int) else None
                # Clients reach the model through the router, not the engine, so
                # a connection count that watched only the engine port would see
                # none of them. Both are the lane.
                _serve_cache["lane_ports"] = lanes | {port}
                return host.split(".")[0], port
    except Exception:
        pass
    return (NODE, SERVE_PORTS[0])


def _fetch(url: str, timeout: float = 4.0) -> str | None:
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def _scrape_serve() -> dict | None:
    """Read one /metrics snapshot, handling both engine families.

    SGLang names its series ``sglang:`` and vLLM names them ``vllm:``; a reader
    that knows only one silently reports a healthy serve as absent. Both are
    parsed here for the same reason the lane reader had to be taught both.
    """
    host, port = _serve_origin()
    text = _fetch(f"http://{host}:{port}/metrics")
    if text is None:
        for alt in SERVE_PORTS:
            if alt == port:
                continue
            text = _fetch(f"http://{host}:{alt}/metrics")
            if text is not None:
                break
    if text is None:
        return None

    gauges = {
        "sglang:num_running_reqs": "running",
        "vllm:num_requests_running": "running",
        "sglang:num_queue_reqs": "queued",
        "vllm:num_requests_waiting": "queued",
        "sglang:full_token_usage": "pool",
        "vllm:kv_cache_usage_perc": "pool",
        "sglang:hicache_host_used_tokens": "host_used",
        "sglang:hicache_host_total_tokens": "host_total",
    }
    counters = {
        "sglang:inter_token_latency_seconds_count": "decode_steps",
        "sglang:prompt_tokens_total": "prompt_tok",
        "sglang:uncached_prompt_tokens_histogram_sum": "uncached_tok",
        # Input tokens the engine served from a cached prefix instead of
        # computing, summed over its device and host tiers. This is the figure
        # the cost split turns on: the remainder (prompt - cached) is what a
        # metered provider would have charged the full prompt rate for.
        "sglang:cached_tokens_total": "cached_tok",
        "sglang:hicache_dropped_tokens_total": "dropped_tok",
        "sglang:generation_tokens_total": "gen_tok",
        "vllm:generation_tokens_total": "gen_tok",
        "vllm:prompt_tokens_total": "prompt_tok",
        # vLLM has counted this in tokens rather than blocks since 0.9; on an
        # older build it under-reports, which shows up as a cost that is too
        # high rather than one that is too low.
        "vllm:prefix_cache_hits_total": "cached_tok",
    }
    out: dict[str, float] = {}
    buckets: dict[str, dict[float, float]] = {"itl": {}, "ttft": {}, "prompt": {}}
    active = 0.0
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        head, _, value = line.rpartition(" ")
        try:
            val = float(value)
        except ValueError:
            continue
        if val != val:  # NaN
            continue
        name = head.split("{", 1)[0]
        if name in gauges:
            out.setdefault(gauges[name], val)
        elif name in counters:  # summed: split by streaming label
            key = counters[name]
            out[key] = out.get(key, 0.0) + val
        elif name.endswith("_bucket"):
            # "uncached_prompt_tokens_histogram" contains "prompt_tokens_
            # histogram", so it must be tested first or every uncached bucket
            # lands in the wrong series and the two medians become one.
            which = (
                "itl"
                if "inter_token_latency" in name
                else "ttft"
                if "time_to_first_token" in name
                else None
                if "uncached_prompt_tokens" in name
                else "prompt"
                if "prompt_tokens_histogram" in name
                else None
            )
            if which and 'le="' in head:
                edge = head.split('le="', 1)[1].split('"', 1)[0]
                with contextlib.suppress(ValueError):
                    buckets[which][float(edge)] = (
                        buckets[which].get(float(edge), 0.0) + val
                    )
        elif "http_requests_active" in name and "/v1/messages" in head:
            active += val
    if not out:
        return None
    out["active"] = active
    out["itl_buckets"] = buckets["itl"]
    out["ttft_buckets"] = buckets["ttft"]
    out["prompt_buckets"] = buckets["prompt"]
    out["at"] = time.monotonic()
    return out


def _median_from_buckets(
    now: dict[float, float], before: dict[float, float]
) -> float | None:
    """Median of a cumulative histogram over one window, by interpolation.

    The buckets are cumulative counts, and the cumulative TOTALS carry every
    request the serve has ever handled -- including, on this deployment, a
    pre-repair era with a seven-times worse profile. Differencing against the
    previous read is what makes the figure describe now.
    """
    edges = sorted(now)
    delta = [(e, now[e] - before.get(e, 0.0)) for e in edges]
    total = delta[-1][1] if delta else 0.0
    if total < 1:
        return None
    half = total / 2.0
    prev_edge, prev_cum = 0.0, 0.0
    for edge, cum in delta:
        if cum >= half:
            span = cum - prev_cum
            if span <= 0:
                return edge
            frac = (half - prev_cum) / span
            return prev_edge + frac * (edge - prev_edge)
        prev_edge, prev_cum = edge, cum
    return delta[-1][0]


def serve_state() -> dict | None:
    """Throttled serve health: gauges now, rates and medians over the window.

    The serve's own card count and context window are resolved as a side
    effect, which the header and the environment band read; the panel's own
    figures come from the recorder's live document.
    """
    nowt = time.monotonic()
    if _serve_cache["raw"] is None or nowt - _serve_cache["at"] >= SERVE_REFRESH:
        fresh = _scrape_serve()
        if fresh is not None:
            hist = _serve_cache["history"]
            hist.append(fresh)
            # Keep one sample older than the window so there is always a base
            # to difference against, then drop the rest.
            while len(hist) > 2 and fresh["at"] - hist[1]["at"] >= SERVE_WINDOW:
                hist.pop(0)
            _serve_cache["raw"] = fresh
            _serve_cache["at"] = nowt
            base = hist[0] if len(hist) > 1 else None
            if base is not None:
                span = fresh["at"] - base["at"]
                if span > 0.5:
                    d_steps = fresh.get("decode_steps", 0.0) - base.get(
                        "decode_steps", 0.0
                    )
                    d_prompt = fresh.get("prompt_tok", 0.0) - base.get(
                        "prompt_tok", 0.0
                    )
                    d_unc = fresh.get("uncached_tok", 0.0) - base.get(
                        "uncached_tok", 0.0
                    )
                    d_out = fresh.get("gen_tok", 0.0) - base.get("gen_tok", 0.0)
                    d_drop = fresh.get("dropped_tok", 0.0) - base.get(
                        "dropped_tok", 0.0
                    )
                    # A serve restarted inside the window leaves every counter
                    # below its base. Differencing through that produces a
                    # negative rate, and clamping it to zero reports an idle
                    # lane that is in fact running. Start the window again.
                    if min(d_prompt, d_out, d_steps) < 0:
                        _serve_cache["history"] = [fresh]
                        _serve_cache["rates"] = None
                        return _serve_state_tail(nowt)
                    itl = _median_from_buckets(
                        fresh["itl_buckets"], base["itl_buckets"]
                    )
                    ttft = _median_from_buckets(
                        fresh["ttft_buckets"], base["ttft_buckets"]
                    )
                    med_in = _median_from_buckets(
                        fresh["prompt_buckets"], base["prompt_buckets"]
                    )
                    prior = _serve_cache["rates"] or {}
                    # An engine with nothing running that also produced nothing
                    # across the window is IDLE, which is a measured state and
                    # not a measurement gap. The two must not share a code
                    # path: carrying forward here is what left a drained lane
                    # reporting 36 tok/s aggregate and 36.4 median/worker
                    # against zero workers, zero running and zero queued --
                    # arithmetic that cannot describe any moment. Carry only
                    # when the window HAD work whose samples merely had not
                    # completed yet.
                    idle = (
                        d_out <= 0.0
                        and d_steps <= 0.0
                        and (fresh.get("running") or 0.0) < 1.0
                        and (fresh.get("queued") or 0.0) < 1.0
                    )

                    def keep(key, value):
                        return (
                            value
                            if value is not None
                            else (None if idle else prior.get(key))
                        )

                    _serve_cache["rates"] = {
                        "idle": idle,
                        "aggregate": max(0.0, d_out or d_steps) / span,
                        # Prompt tokens the engine actually had to compute.
                        # The arriving rate is not the card's prefill load and
                        # on this deployment overstates it fiftyfold, because
                        # almost all of it is answered from cache at no cost.
                        "prefill": max(0.0, d_unc) / span,
                        # The share of prefill the engine did not have to
                        # compute. Counted in tokens rather than requests, so a
                        # handful of long cold prompts cannot hide behind many
                        # short warm ones.
                        "median_in": keep("median_in", med_in),
                        "dropped": max(0.0, d_drop) * 60.0 / span,
                        "hit": keep(
                            "hit",
                            (1.0 - d_unc / d_prompt) * 100.0 if d_prompt > 0 else None,
                        ),
                        "per_worker": keep(
                            "per_worker", (1.0 / itl) if itl and itl > 0 else None
                        ),
                        "ttft": keep("ttft", ttft),
                    }
        elif nowt - _serve_cache["at"] >= SERVE_REFRESH * 4:
            _serve_cache["raw"] = None
            _serve_cache["rates"] = None
            _serve_cache["history"] = []
    return _serve_state_tail(nowt)


def _serve_state_tail(nowt: float) -> dict | None:
    """Model card and the state the panel renders from.

    The serve's own card count and context window are resolved here, which the
    serve header and the environment band read. Probing for the model card is
    kept to its own refresh so it does not run on every tick.
    """
    if _serve_cache["raw"] is None:
        return None
    if (
        _serve_cache["model"] is None
        or nowt - _serve_cache["model_at"] >= MODEL_REFRESH
    ):
        host, port = _serve_origin()
        body = _fetch(f"http://{host}:{port}/v1/models", timeout=3.0)
        if body:
            try:
                import json

                card = (json.loads(body).get("data") or [{}])[0]
                ctx = card.get("max_model_len")
                _serve_cache["model"] = (
                    card.get("id", "?"),
                    f"{ctx // 1000}k" if isinstance(ctx, int) and ctx >= 1000 else "",
                )
            except Exception:
                _serve_cache["model"] = ("?", "")
            _serve_cache["model_at"] = nowt
    return {
        "raw": _serve_cache["raw"],
        "rates": _serve_cache["rates"],
        "model": _serve_cache["model"],
    }


_live_cache: dict = {"at": 0.0, "doc": None}


def _read_live() -> dict | None:
    """One ``agent watch --json --live`` reading, or None if it did not answer.

    The live document is the recorder's own account of the lane and the receipt
    tail -- both published sources -- so this is a read of what was written, not
    a probe of the serve. A non-zero exit, unparseable output or a timeout is
    reported as absence, never as zeros.
    """
    import json

    try:
        proc = subprocess.run(
            [IMAS_AMBIX_CLI, "agent", "watch", "--json", "--live"],
            capture_output=True,
            text=True,
            timeout=LIVE_TIMEOUT,
        )
        if proc.returncode != 0:
            return None
        doc = json.loads(proc.stdout)
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def _watch_live() -> dict | None:
    """The live document, refreshed at the panel's own cadence.

    A read that fails keeps the last document for a short grace so a single
    missed call does not blank the panel, then clears it: a stale document shown
    as current is the reading this panel exists to replace.
    """
    nowt = time.monotonic()
    if _live_cache["doc"] is not None and nowt - _live_cache["at"] < SERVE_REFRESH:
        return _live_cache["doc"]
    doc = _read_live()
    if doc is not None:
        _live_cache["doc"], _live_cache["at"] = doc, nowt
        return doc
    if _live_cache["doc"] is not None and nowt - _live_cache["at"] < SERVE_REFRESH * 4:
        return _live_cache["doc"]
    _live_cache["doc"] = None
    return None


def _ctx_label() -> str:
    """The serve's context window, when the ledger reading already resolved it."""
    model = _serve_cache.get("model") or ("", "")
    return model[1] if len(model) > 0 and model[1] else ""


CHECKPOINT_REFRESH = 30.0  # seconds between endpoint-document reads for the checkpoint
_checkpoint_cache: dict = {"at": float("-inf"), "label": None}


def _checkpoint_label() -> str | None:
    """The checkpoint the served engine loaded, from the published endpoint document.

    The served name stays the same across a weights-only rotation, so it cannot
    say which checkpoint answered; the endpoint document's ``checkpoints`` map
    can, keyed by engine id. It is shown as the repository's last path part and
    a short commit. None means the document records no checkpoint for any
    engine, which the header states rather than leaving blank.
    """
    import json

    nowt = time.monotonic()
    if nowt - _checkpoint_cache["at"] < CHECKPOINT_REFRESH:
        return _checkpoint_cache["label"]
    label = None
    try:
        with open(ENDPOINT_DOC, encoding="utf-8") as fh:
            checkpoints = json.load(fh).get("checkpoints") or {}
        names = []
        for value in checkpoints.values():
            repo, _, commit = str(value).partition("@")
            short = commit[:8] if commit and commit != "unpinned" else commit
            names.append(f"{repo.rsplit('/', 1)[-1]}@{short}" if short else repo)
        label = ", ".join(sorted(set(names))) or None
    except (OSError, ValueError, AttributeError):
        label = None
    _checkpoint_cache.update(at=nowt, label=label)
    return label


def serve_head(lane: dict, live: dict, gpu_count: int) -> str:
    """The lane in one line: what is served, on what, and at what context.

    Everything here is constant for the life of the deployment, which is why
    it is here and not in the grid: a figure that cannot change during a
    session is not a reading, and repeating it every second costs ink that
    never says anything new. Stated once, it makes the grid's numbers
    interpretable -- the model names what was served, and the context window
    is the ceiling every prompt below is measured against.

    The served model comes from the recorder's live block, not from a probe of
    the engine: the same document the grid below is read from describes what
    was served, so the header and the grid can never name two different
    serves. The listed price per million the header used to print came from
    the private price fetch, which is gone; the lane's cost now lives, with
    every other serving figure, in the period ledger below.
    """
    model = live.get("served_name") or "?"
    bits = [f"[bold]{model}[/]"]
    checkpoint = _checkpoint_label()
    bits.append(
        f"[cyan]{checkpoint}[/]" if checkpoint else "[yellow]checkpoint unrecorded[/]"
    )
    if gpu_count:
        bits.append(f"[dim]{gpu_count}×H200[/]")
    ctx = _ctx_label()
    if ctx:
        bits.append(f"[dim]{ctx} ctx[/]")
    # The stale mark belongs here rather than in the grid: every figure copied
    # from the lane document now sits in the grid's last band, so one mark on
    # the header says the same thing once instead of marking three rows that
    # all fail together.
    stale_age = _lane_age_stale(lane)
    if stale_age is not None:
        bits.append(_stale_mark(stale_age))
    return " [dim]·[/] ".join(bits)


def _slots_headroom(lane: dict) -> float | None:
    """The lane's own headroom in requests -- the figure orchestrators read.

    The top-level ``headroom`` is the admission-clamped figure; the engine's own
    headroom travels beside it as ``engine_headroom``. Reporting the engine's
    here would tell a dispatcher there is room admission is not offering, which
    is the reading that made orchestrators over-dispatch before the clamp.
    """
    value = lane.get("headroom")
    return value if isinstance(value, (int, float)) else None


def _slots_worker_count(lane: dict) -> int | None:
    """The lane's worker-slot count, from the nested object the router publishes.

    The lane document's ``worker_slots`` is an object -- ``live_runs``,
    ``requests_per_run``, ``worker_slots`` and ``unkeyed_share`` -- rather than
    a bare number, so the count is the nested ``worker_slots.worker_slots``.
    A lane that never published the object, or published it without the count,
    leaves the cell a dash rather than a zero, because a zero is a measurement
    the lane never made.
    """
    block = lane.get("worker_slots")
    if not isinstance(block, dict):
        return None
    value = block.get("worker_slots")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _lane_age_stale(lane: dict) -> float | None:
    """The lane document's age in seconds when it is past its own shelf life.

    The router republishes the lane document on its own cadence and stamps it
    with the age its readers should tolerate (``suggested_shelf_life_seconds``).
    A document past that shelf life is not a weaker reading of the lane -- it is
    a reading of a lane that may no longer exist, so every figure copied from it
    is marked rather than shown as current. The router may also say so directly
    (``stale``); either sign carries the age. A document that publishes neither
    is not called stale on a guess.
    """
    age = lane.get("age_seconds")
    shelf = lane.get("suggested_shelf_life_seconds")
    if lane.get("stale") is True:
        return age if isinstance(age, (int, float)) else -1.0
    if (
        isinstance(age, (int, float))
        and isinstance(shelf, (int, float))
        and age > shelf
    ):
        return age
    return None


def _stale_mark(age: float) -> str:
    """The visible mark printed on a row built from a stale lane document."""
    when = f"{age:,.0f}s" if age >= 0 else "age unknown"
    return f"[bold yellow]stale[/] [yellow]{when}[/]"


def _ramp(value: float | None, good: float, fair: float, inverted: bool = False) -> str:
    """Health as colour, which is the only mark here that is not a digit."""
    if value is None:
        return "dim"
    if inverted:
        return "green" if value <= good else "yellow" if value <= fair else "red"
    return "green" if value >= good else "yellow" if value >= fair else "red"


def pct(
    value: float | None, mode: str = "fill", places: int = 0, width: int = 5
) -> str:
    """A percentage as a filled bar and its figure, the same way every time.

    Every proportion on this panel is drawn like this -- cache hit, pool
    occupancy, host-tier share, power against cap, cached share in the ledger
    -- because a rule applied to some percentages and not others has to be
    learnt, while one applied to all of them is simply read. The bar carries
    the magnitude at a glance and the figure carries it exactly; the pair is
    worth the ink only while it is universal.

    ``mode`` sets what the colour means. ``good`` is high-is-healthy, ``warn``
    is high-is-trouble, and ``fill`` is neither -- an occupancy that is merely
    a fact, coloured by how full it is rather than by whether it should be.
    """
    if value is None:
        return "—"
    v = max(0.0, min(100.0, value))
    fill = round(v / 100 * width)
    if mode == "good":
        col = _ramp(v, 80, 40)
    elif mode == "warn":
        col = _ramp(v, 80, 95, inverted=True)
    else:
        col = "red" if v >= 90 else "magenta" if v >= 70 else "cyan"
    bars = "█" * fill + "░" * (width - fill)
    return f"[{col}]{bars}[/] {value:.{places}f}"


def serve_grid(
    lane: dict, live: dict, env: dict | None, idx: set[int] | None, tight: bool = False
) -> Table:
    """Every live reading, in three bands on the ledger's own column grid.

    Three bands, distinguished by weight alone rather than by borders, gauges
    or rules: the engine's own throughput, the cards it runs on with the pool
    they hold, and the router's gate with the room left to dispatch into. Each
    band is one label row over one value row with a blank separator between
    bands, so the whole grid takes at most eight rows. The figures share the
    ledger's seven columns -- the first 13 characters wide and the rest 12 --
    so a reader scanning down does not have to re-find the columns, and the
    figures are grouped by what they describe rather than one per column.

    Read from the recorder's ``lane`` and ``live`` blocks, never from a scrape
    of the engine: the engine answers only for the requests that reached it, so
    it cannot see the requests the generation gate is holding or the headroom
    orchestrators dispatch against, and it stops answering at all once the
    engine binds loopback.
    """
    cards = {
        k: v for k, v in (env or {}).items() if idx is None or not idx or int(k) in idx
    }
    temps = [v[0] for v in cards.values() if v[0] is not None]
    power = [v[1] for v in cards.values() if v[1] is not None]
    caps = [v[2] for v in cards.values() if v[2] is not None]

    # box, padding and pad_edge are the ledger's own settings verbatim, so the
    # bands and the ledger below share one column pitch and a reader scanning
    # down never has to re-find the columns. show_header is off because the
    # bands label their own rows; the box is the ledger's so the geometry is
    # identical.
    t = Table(
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        expand=False,
        pad_edge=False,
        show_header=False,
    )
    for i in range(7):
        t.add_column("", justify="right", width=13 if i == 0 else 12, no_wrap=True)

    def band(labels: list[str], values: list[str], style: str) -> None:
        """Label row over value row, with a blank line BETWEEN bands only.

        Trailing it instead leaves a gap under the last band that the box's own
        padding then doubles.
        """
        if t.row_count and not tight:
            t.add_row(*[""] * 7)
        pad = [""] * (7 - len(labels))
        t.add_row(*(labels + pad), style="dim")
        t.add_row(*(values + pad), style=style)

    dash = "—"
    mean_running = live.get("mean_running")
    queued = live.get("engine_queue")
    agg = live.get("generation_toks_per_s")
    per = live.get("per_stream_toks_per_s")
    prefill = live.get("prefill_toks_per_s")
    ttft = live.get("median_time_to_first_token_s")
    hit = live.get("cache_hit_rate")
    measured = live.get("state") == "measured"
    # The first band is what the lane is judged on. "running" is the count of
    # requests the engine holds, a measured quantity; "workers" was that same
    # count under a name that claimed to count processes, and router receipts
    # over 15 minutes carried 14 distinct run ids against 6.6 mean requests in
    # flight -- a worker holds a request about half the time, so the two are not
    # the same population and only the request count is measured; the lane's
    # own worker count lives on the section's slots row below. "prefill tok/s"
    # is shortened to "prefill/s" because it is a character wider than the
    # grid's 12-character columns, and the first column is the request count.
    band(
        [
            "running",
            "queued",
            "tok/s out",
            "tok/s each",
            "prefill/s",
            "ttft s",
            "cache %",
        ],
        [
            f"[cyan]{mean_running:.1f}[/]"
            if isinstance(mean_running, (int, float))
            else dash,
            (
                f"[red]{queued:.0f}[/]"
                if isinstance(queued, (int, float)) and queued >= 1
                else "[dim]0[/]"
                if isinstance(queued, (int, float))
                else dash
            ),
            fmt_tokens(agg)
            if measured and agg is not None and not isinstance(agg, str)
            else dash,
            (
                f"[{_ramp(100.0 * per / PER_WORKER_REF, 60, 30)}]{per:.1f}[/]"
                if measured and per
                else dash
            ),
            fmt_tokens(prefill) if measured and prefill is not None else dash,
            f"{ttft:.1f}"
            if measured and ttft is not None and not isinstance(ttft, str)
            else dash,
            pct(100.0 * hit, "good") if measured and hit is not None else dash,
        ],
        "bold",
    )

    # The conditions the engine runs under, and the pool it holds. Pool
    # occupancy sits here with the cards rather than with the throughput above
    # because it describes the same hardware.
    pool = live.get("pool_occupancy")
    if temps and power and caps:
        cap, draw = sum(caps), sum(power)
        # An H200 throttles around the mid-eighties, so the ramp warns before
        # the clocks come down rather than after.
        # The hottest card is the one that throttles, so the mean is not shown.
        card_vals = [
            f"[{_ramp(max(temps), 75, THROTTLE_C, inverted=True)}]{max(temps):.0f}[/]",
            f"{draw / 1000.0:.2f}",
            pct(100.0 * draw / cap, "warn"),
        ]
    else:
        card_vals = [dash, dash, dash]
    band(
        ["°C max", "kW", "% of cap", "kv pool %"],
        card_vals + [pct(100.0 * pool) if pool is not None else dash],
        "",
    )

    # The router's own gate, and the room left to dispatch into. A request held
    # here never reaches the engine, so the engine's queue reads zero while the
    # gate is holding work -- reading only the engine's queue cannot tell a
    # drained lane from a full one. In flight is shown over the width in force
    # (``effective_width``), not the configured width the gate may be shrinking
    # from. The oldest wait and the verdict are published only in the admission
    # block. Available % is the lane's own ``availability_percent``, published
    # by the gate: the panel computes nothing from headroom or width itself, so
    # its figure and the lane document's can never disagree.
    gate = lane.get("router_generation_gate") or {}
    admission = lane.get("admission") or {}
    width = gate.get("effective_width")
    in_flight = gate.get("in_flight")
    held = gate.get("waiting")
    oldest = admission.get("oldest_wait_seconds")
    verdict = admission.get("verdict")
    headroom = _slots_headroom(lane)
    slots = _slots_worker_count(lane)
    availability = lane.get("availability_percent")
    in_flight_s = (
        (f"{in_flight}" if isinstance(in_flight, int) else dash)
        + "/"
        + (f"{width}" if isinstance(width, int) else dash)
    )
    band(
        [
            "in flight",
            "router queue",
            "oldest wait",
            "verdict",
            "headroom",
            "worker slots",
            "available %",
        ],
        [
            in_flight_s,
            (
                f"[red]{held}[/]"
                if isinstance(held, int) and held >= 1
                else f"[dim]{held}[/]"
                if isinstance(held, int)
                else dash
            ),
            fmt_span(oldest) if isinstance(oldest, (int, float)) else dash,
            f"[bold]{verdict}[/]" if verdict else dash,
            f"[bold]{headroom}[/]" if isinstance(headroom, (int, float)) else dash,
            f"{slots}" if slots is not None else dash,
            pct(availability, "good")
            if isinstance(availability, (int, float))
            and not isinstance(availability, bool)
            else dash,
        ],
        "",
    )

    return t


CALLER_ROWS = 8  # callers listed; the rest fold into one row
OWNER_RETRY = 600.0  # seconds before an unmatched session is looked up again
HOST_USERS_TTL = 60.0  # seconds a node's job owners are trusted
CLAUDE_PROJECTS = os.path.expanduser("~/.claude/projects")
RECKON_MOUNTS = os.path.expanduser("~/.config/reckon/mounts.json")
_DATED_SESSION = re.compile(r"^(?P<project>.+)-\d{8}$")


class _Resolver:
    """Names for the callers table, looked up on a worker thread.

    Every lookup here can stall: a reverse DNS query, a glob over the
    transcript directory on GPFS (about 2 s), an squeue round trip. The render
    loop asks for a name and gets whatever is already known -- None the first
    time -- while the worker fills it in, so a slow lookup costs one refresh of
    a placeholder rather than a frozen screen.
    """

    def __init__(self) -> None:
        import queue
        import threading

        self._lock = threading.Lock()
        self._known: dict[tuple[str, str], tuple[float, object]] = {}
        self._pending: set[tuple[str, str]] = set()
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def get(
        self, kind: str, key: str, *, hit_ttl: float | None, miss_ttl: float
    ) -> object:
        """The known value, queuing a lookup when it is absent or expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._known.get((kind, key))
            if entry is not None:
                at, value = entry
                ttl = hit_ttl if value is not None else miss_ttl
                if ttl is None or now - at < ttl:
                    return value
            if (kind, key) not in self._pending:
                self._pending.add((kind, key))
                self._queue.put((kind, key))
            return entry[1] if entry is not None else None

    def _work(self) -> None:
        while True:
            kind, key = self._queue.get()
            try:
                value = getattr(self, f"_lookup_{kind}")(key)
            except Exception:
                value = None
            with self._lock:
                self._known[(kind, key)] = (time.monotonic(), value)
                self._pending.discard((kind, key))

    @staticmethod
    def _lookup_node(address: str) -> str | None:
        """The SLURM node name behind an address, which keeps its site prefix."""
        import socket

        return socket.gethostbyaddr(address)[0].split(".")[0] or None

    @staticmethod
    def _lookup_session(session: str) -> dict | None:
        """This account's project for a coordinator session, or None.

        A transcript under ~/.claude/projects is the session itself, so its
        directory names the project and proves the session ran in this
        account. A reckon session named ``<project>-<YYYYMMDD>`` is matched
        against this account's reckon mounts instead, which is how a
        coordinator running on another machine names itself.
        """
        import glob
        import json

        pattern = os.path.join(CLAUDE_PROJECTS, "*", glob.escape(session) + ".jsonl")
        found = glob.glob(pattern)
        if found:
            folder = os.path.basename(os.path.dirname(found[0]))
            home = os.path.expanduser("~").replace("/", "-")
            rest = folder.removeprefix(home).lstrip("-").removeprefix("Code-")
            return {"project": rest.split("--")[0] or "~", "basis": "transcript"}
        dated = _DATED_SESSION.match(session)
        if dated:
            with open(RECKON_MOUNTS, encoding="utf-8") as fh:
                mounts = json.load(fh)
            if dated["project"] in mounts:
                return {"project": dated["project"], "basis": "reckon project"}
        return None

    @staticmethod
    def _lookup_users(node: str) -> tuple[str, ...] | None:
        """Accounts holding jobs on *node*; None when it is not a SLURM node."""
        proc = subprocess.run(
            ["squeue", "-w", node, "-h", "-o", "%u"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        return tuple(sorted({u.strip() for u in proc.stdout.split() if u.strip()}))


_resolver: _Resolver | None = None


def _names() -> _Resolver:
    global _resolver
    if _resolver is None:
        _resolver = _Resolver()
    return _resolver


def _short_session(session: str) -> str:
    dated = _DATED_SESSION.match(session)
    return dated["project"] if dated else session[:8]


def _caller_owner(caller: dict) -> tuple[str, str]:
    """The row's label and the account it belongs to, as far as can be shown."""
    names = _names()
    session = caller.get("session")
    if session:
        match = names.get("session", session, hit_ttl=None, miss_ttl=OWNER_RETRY)
        if isinstance(match, dict):
            return str(match["project"]), USER
        return _short_session(session), "?"
    hosts = caller.get("hosts") or []
    address = hosts[0] if hosts else ""
    node = names.get("node", address, hit_ttl=None, miss_ttl=OWNER_RETRY)
    users = (
        names.get("users", str(node), hit_ttl=HOST_USERS_TTL, miss_ttl=HOST_USERS_TTL)
        if node
        else None
    )
    label = "no session"
    if not isinstance(users, tuple) or not users:
        return label, "?"
    if users == (USER,):
        return label, USER
    if USER in users:
        return label, "?"
    return label, users[0] if len(users) == 1 else f"{users[0]}+{len(users) - 1}"


def callers_table(block: dict | None) -> object | None:
    """Whose requests fill the gate: one row per caller, heaviest first.

    ``in use`` is the caller's time-averaged requests occupying the gate,
    waiting or generating, over the block's window: the unit the gate's width
    is counted in, and capacity the caller is using, not work held back. The
    router publishes it as ``held``; the header names what it measures, because
    "held" reads as queued. ``per run`` is the fan-out a run-counted share
    cannot see: a worker running parallel subagents occupies several requests
    while counting as one run, so anything above 1 is coloured. ``wait s`` is
    the mean time a caller's requests spent queued at the gate, and the only
    column that measures waiting.
    """
    if not isinstance(block, dict):
        return None
    if block.get("state") != "measured":
        detail = block.get("detail") or "no reading"
        return Text.from_markup(f"[yellow]callers unavailable[/][dim] — {detail}[/]")
    callers = block.get("callers") or []
    window = block.get("window_seconds")
    head = (
        f"callers {window / 60:.0f}m" if isinstance(window, (int, float)) else "callers"
    )
    if not callers and block.get("covered") is not False:
        # The block counts finished requests only, so an empty one is stated in
        # those words: the gate may still hold requests that have not ended.
        return Text.from_markup(f"[dim]{head}   no request finished in the window[/]")
    t = Table(
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        expand=False,
        pad_edge=False,
        header_style="dim",
    )
    for i, name in enumerate(
        (head, "user", "host", "runs", "in use", "per run", "wait s")
    ):
        t.add_column(
            name,
            justify="left" if i < 3 else "right",
            width=13 if i == 0 else 12,
            no_wrap=True,
        )
    names = _names()
    dash = "—"
    shown, folded = callers[:CALLER_ROWS], callers[CALLER_ROWS:]
    owners = [_caller_owner(caller) for caller in shown]
    # Two sessions in one project would read as one caller listed twice, so a
    # repeated session label carries the session's first characters.
    repeats = {
        label
        for label, _ in owners
        if sum(1 for other, _ in owners if other == label) > 1
    }
    for caller, (label, owner) in zip(shown, owners, strict=False):
        session = caller.get("session")
        if session and label in repeats:
            label = f"{label[:8]} {session[:4]}"
        hosts = caller.get("hosts") or []
        node = (
            names.get("node", hosts[0], hit_ttl=None, miss_ttl=OWNER_RETRY)
            if hosts
            else None
        )
        # The site prefix is on every node name, so it carries nothing in a
        # 12-character column; the SLURM lookups above keep the full name.
        host_s = (
            str(node).removeprefix("98dci4-") if node else (hosts[0] if hosts else dash)
        )
        if len(hosts) > 1:
            host_s = f"{host_s}+{len(hosts) - 1}"
        per = caller.get("held_per_run")
        wait = caller.get("mean_gate_wait_s")
        held = caller.get("held")
        t.add_row(
            label,
            owner if owner == USER else f"[yellow]{owner}[/]",
            host_s,
            f"{caller.get('runs') or 0}" if caller.get("session") else dash,
            f"{held:.1f}" if isinstance(held, (int, float)) else dash,
            (
                f"[{_ramp(per, 1.0, 2.0, inverted=True)}]{per:.1f}[/]"
                if isinstance(per, (int, float))
                else dash
            ),
            (
                f"[{_ramp(wait, 5.0, 30.0, inverted=True)}]{wait:.0f}[/]"
                if isinstance(wait, (int, float))
                else dash
            ),
        )
    if folded:
        held = sum(c.get("held") or 0.0 for c in folded)
        t.add_row(
            f"+{len(folded)} more", "", "", "", f"{held:.1f}", "", "", style="dim"
        )
    if not callers:
        t.add_row("[dim]idle[/]", "", "", "", "", "", "")
    if block.get("covered") is False:
        return Group(
            t,
            Text.from_markup(
                "[yellow]window cut short[/][dim] — the record tail reached its "
                "read bound before the window's start[/]"
            ),
        )
    return t


def prime_callers() -> None:
    """Queue every caller's name lookups now, so a one-shot frame has them.

    The one-shot reading spends its whole window sampling the cards before it
    renders, which is ample time for the lookups to finish behind it; queued at
    render time instead, every row of the single frame would be a placeholder.
    """
    doc = _watch_live() or {}
    for caller in (doc.get("callers") or {}).get("callers") or []:
        _caller_owner(caller)


def _document_age(published_at: object) -> float | None:
    """Age in seconds of the published document, from its own UTC stamp.

    The publisher stamps the document to the second; an absent or malformed
    stamp is reported as unknown rather than treated as fresh, because a
    document whose age cannot be established has not been shown to be current.
    """
    if not isinstance(published_at, str):
        return None
    import datetime

    try:
        when = datetime.datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    when = when.replace(tzinfo=datetime.UTC)
    now = datetime.datetime.now(datetime.UTC)
    return max(0.0, (now - when).total_seconds())


def _watch_document() -> tuple[dict | None, str | None]:
    """The fleet's published watch document, with a note when it warrants one.

    Returns ``(doc, note)``. ``doc`` is ``None`` with a note when the document
    is missing or unparsable; it is the document itself with an age note when
    the document is present but old. Every figure the period ledger prints
    comes from ``doc``; the note is the only thing this panel adds of its own.
    """
    import json

    try:
        with open(WATCH_DOC, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None, (f"[yellow]watch document unavailable[/][dim] — {WATCH_DOC}[/]")
    if not isinstance(doc, dict):
        return None, (f"[yellow]watch document unparsable[/][dim] — {WATCH_DOC}[/]")
    age = _document_age(doc.get("published_at"))
    if age is not None and age > LEDGER_STALE_AFTER:
        return doc, (
            f"[yellow]stale[/] [yellow]{fmt_span(age)}[/]"
            f"[dim] — published {age:,.0f}s ago[/]"
        )
    return doc, None


def serve_ledger_table() -> object | None:
    """Tokens and their open-market cost, one row per published period.

    Every figure here is rendered from the fleet's published watch document,
    which the recorder derives from the lane's own receipts, so a period's
    tokens and cost can never disagree with the document the rest of the fleet
    reads. The ``window`` column is not decoration: a day's row backed by six
    hours of record is a real answer to a different question, and without that
    column it is indistinguishable from a quiet day. A missing or unparsable
    document is stated in the panel; a document older than
    LEDGER_STALE_AFTER still shows its figures, labelled with its age.
    """
    doc, note = _watch_document()
    if doc is None:
        return Text.from_markup(note or "")
    rows = doc.get("ledger") or []
    if not rows:
        return Text.from_markup(
            f"[yellow]watch document carries no ledger rows[/][dim] — {WATCH_DOC}[/]"
        )

    def _full(p: dict) -> bool:
        period = float(p.get("period") or 0.0)
        return bool(period) and float(p.get("covered") or 0.0) >= 0.98 * period

    # A period backed by its whole record needs no coverage figure; the column
    # is drawn only when some row answers for less than its period.
    partial = not all(_full(p) for p in rows)
    heads = ["ledger", "tok in", "cached %", "tok out", "util %", "cost $"]
    if partial:
        heads.append("covered")
    t = Table(
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        expand=False,
        pad_edge=False,
        header_style="dim",
    )
    for i, head in enumerate(heads):
        t.add_column(head, justify="right", width=13 if i == 0 else 12, no_wrap=True)
    for p in rows:
        tokens_in = float(p.get("tokens_in") or 0.0)
        cached = float(p.get("cached") or 0.0)
        tokens_out = float(p.get("tokens_out") or 0.0)
        util = p.get("utilisation")
        period = float(p.get("period") or 0.0)
        covered = float(p.get("covered") or 0.0)
        share = 100.0 * cached / tokens_in if tokens_in > 0 else 0.0
        cost = p.get("cost")
        cost_s = "—" if cost is None else fmt_usd(cost).lstrip("$")
        covered_s = "full" if period and covered >= 0.98 * period else fmt_span(covered)
        cells = [
            str(p.get("label") or "?"),
            fmt_tokens(tokens_in),
            pct(share, "good"),
            fmt_tokens(tokens_out),
            pct(float(util) if isinstance(util, (int, float)) else None),
            cost_s,
        ]
        if partial:
            cells.append(covered_s)
        t.add_row(*cells, style="dim")
    if note is not None:
        return Group(t, Text.from_markup(note))
    return t


def serve_table(
    live_doc: dict | None,
    gpu_count: int,
    env: dict | None = None,
    env_idx: set[int] | None = None,
    tight: bool = False,
) -> Group:
    """The served lane as a dashboard, read from the recorder's documents.

    The live panel is ``agent watch --json --live``: the ``lane`` block copied
    from the router's published lane document and the ``live`` block differenced
    from the recorder's newest raw receipt file. The period ledger below it is
    the fleet's published watch document. Both are published sources, so this
    panel needs neither a scrape of the engine nor an index read.
    """
    if not live_doc:
        return Group(
            Text.from_markup(
                "[yellow]live watch unavailable[/][dim] — "
                "agent watch --json --live returned nothing[/]"
            )
        )

    lane = live_doc.get("lane") or {}
    live = live_doc.get("live") or {}

    parts = [
        serve_head(lane, live, gpu_count),
        serve_grid(lane, live, env, env_idx, tight),
    ]
    callers = callers_table(live_doc.get("callers"))
    if callers is not None:
        parts.append(callers)
    ledger = serve_ledger_table()
    if ledger is not None:
        parts.append(ledger)
    return Group(*parts)


def node_state() -> str:
    """Node availability from sinfo (idle / mix / alloc / drain / down)."""
    out = _run(["sinfo", "-n", NODE, "-h", "-o", "%t %E"]).strip()
    return out or "unknown"


def overlap_steps() -> tuple[list[str], str | None]:
    """Our running steps on NODE (the only ones --overlap can attach to) and
    the reservation group (A/B) they belong to — the one group we can sample."""
    step_rows = [
        ln.split("\t")
        for ln in _run(
            [
                "squeue",
                "-u",
                USER,
                "-w",
                NODE,
                "--states=R",
                "--noheader",
                "-o",
                "%i\t%v",
            ]
        ).splitlines()
        if ln.strip()
    ]
    steps = [r[0] for r in step_rows if r and r[0]]
    our_group = next(
        (_resv_group(r[1]) for r in step_rows if len(r) > 1 and _resv_group(r[1])),
        None,
    )
    return steps, our_group


def node_jobs() -> list[list[str]]:
    """Running jobs on NODE plus pending jobs in the betelgeuse partition.

    Any user, any group — this is where Group B load is visible (its per-GPU
    util is cgroup-hidden). Pending jobs are fetched separately by partition
    since squeue -w only returns jobs that have been assigned to a node.
    """
    fmt = ["-o", "%i|%u|%a|%j|%P|%T|%M|%b|%v"]
    running = [
        ln.split("|")
        for ln in _run(["squeue", "-w", NODE, "--noheader", *fmt]).splitlines()
        if ln
    ]
    pending = [
        ln.split("|")
        for ln in _run(
            ["squeue", "-p", "betelgeuse", "--states=PD", "--noheader", *fmt]
        ).splitlines()
        if ln
    ]
    seen = {(r + [""])[0].strip() for r in running}
    return running + [p for p in pending if (p + [""])[0].strip() not in seen]


def _env_fields(*values: str) -> tuple:
    """Sampler environment fields → floats, with an absent sensor left as None.

    An empty field is a card that did not report, and it must stay absent: a
    missing power reading coerced to zero pulls a fleet total down by a whole
    card and reads as headroom that is not there.
    """
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return tuple(out)


_serve_idx_cache: dict = {"jobid": None, "idx": set()}


def serve_gpu_idx(jobs: list[list[str]], model: str) -> set[int]:
    """Physical cards the serve job holds, so the environment row describes IT.

    A node-wide temperature and power total answers a question about the node;
    what the serve line needs is the cost of running THIS model. The two differ
    by whatever else is co-resident, which on this node is usually a training
    job. The serve job is found by name because that is what the launcher sets
    it to -- punctuation differs between a model id and a SLURM job name, so the
    comparison ignores it. An empty result means the row falls back to every
    card it can see, and says how many that was.
    """
    want = _norm_model(model)
    if not want:
        return set()
    for row in jobs or []:
        r = (list(row) + [""] * 9)[:9]
        if not r[5].strip().upper().startswith("R"):
            continue
        if _norm_model(r[3]) != want:
            continue
        if _serve_idx_cache["jobid"] != r[0]:
            # scontrol per tick would be a subprocess per second for a mapping
            # that cannot change while the job lives.
            _serve_idx_cache["jobid"] = r[0]
            _serve_idx_cache["idx"] = allocated_gpu_idx([r[0]])
        return _serve_idx_cache["idx"]
    return set()


def _parse_idx(spec: str) -> set[int]:
    """Expand a SLURM GRES IDX spec like '0-3,6-7' into {0,1,2,3,6,7}."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                out.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def allocated_gpu_idx(job_ids: list[str]) -> set[int]:
    """Physical GPU indices allocated on NODE to ``job_ids`` (scontrol GRES IDX).

    squeue gives only GPU *counts*; the physical card numbers come from
    ``scontrol show job -d`` (``GRES=gpu:h200:6(IDX:0-3,6-7)`` on the NODE line).
    Used to tell an UNSAMPLED card apart: allocated-but-unreachable (a cross-NUMA
    card of a running job, or another group's) vs genuinely idle/free (no job) —
    so freed cards do not read as 'unreachable' as jobs end.
    """
    idx: set[int] = set()
    for j in job_ids:
        out = _run(["scontrol", "show", "job", str(j), "-d"])
        for line in out.splitlines():
            if NODE in line and "IDX:" in line:
                seg = line.split("IDX:", 1)[1].split(")", 1)[0]
                idx |= _parse_idx(seg)
    return idx


def sample(window: int) -> dict:
    """Sample util ONLY by overlapping our running steps; never submit a job.

    Submitting a 'throwaway' GPU job to read nvidia-smi when nothing of ours is
    running is both wasteful and dangerous: on a drained/busy node those jobs
    queue forever (PENDING) and pile up, and a job that hangs in I/O can drain
    the node (RCA docs/rca-node-drain-2026-05-27b.html). So when we have no
    running step we report node STATE + the full job table only — no per-GPU
    util.

    The job table is node-wide (any user, any partition, any reservation),
    pulled with ``squeue -w NODE``.
    """
    nstate = node_state()
    steps, our_group = overlap_steps()

    # jobid -> human name, so each card is labelled by the PROCESS that owns it
    # (not a fake global gpu index that silently shifts when a sampler drops).
    names: dict[str, str] = {}
    for ln in _run(
        ["squeue", "-u", USER, "-w", NODE, "--states=R", "--noheader", "-o", "%i\t%j"]
    ).splitlines():
        c = ln.split("\t")
        if c and c[0].strip():
            names[c[0].strip()] = (
                c[1].strip() if len(c) > 1 and c[1].strip() else c[0].strip()
            )

    # Sample each owned step in its OWN cgroup, in PARALLEL, and keep the result
    # PER JOB. A job whose sampler returns nothing is recorded with empty gpus
    # (rendered 'unsampled') — never silently dropped, so every running process
    # we own is always visible.
    per_job: list[dict] = []
    node = None
    env: dict[str, tuple] = {}
    card_util: dict[str, float] = {}
    if steps:
        procs = [
            (
                j,
                subprocess.Popen(
                    [
                        "srun",
                        "--jobid",
                        j,
                        "--overlap",
                        "python3",
                        SAMPLER,
                        str(window),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ),
            )
            for j in steps
        ]
        for j, p in procs:
            try:
                out, _ = p.communicate(timeout=window + 90)
            except subprocess.TimeoutExpired:
                p.kill()
                out = ""
            jg = []
            for line in (out or "").splitlines():
                c = line.split("\t")
                if c[0] == "GPU" and len(c) >= 7:
                    jg.append((c[1], *(float(x) for x in c[3:7])))  # phys,un,ua,mn,ma
                    card_util[c[1]] = float(c[4])  # the window's mean util
                    if len(c) >= 10:
                        env[c[1]] = _env_fields(c[7], c[8], c[9])
                elif c[0] == "NODE" and len(c) >= 4 and node is None:
                    node = tuple(float(x) for x in c[1:4])  # cpu,used,total
            per_job.append(
                {
                    "jobid": j,
                    "name": names.get(j, j),
                    "gpus": jg,
                    "sampled": bool((out or "").strip()),
                }
            )
        source = f"{len(steps)} step{'s' if len(steps) != 1 else ''}"
    else:
        source = "no overlappable step (need a running job we own on this node)"

    # Fallback ONLY when no step returned any card (e.g. a serving job that has
    # exhausted its SLURM step slots): surface the vLLM /metrics load so the
    # panel isn't a lying blank.
    vllm = None
    if steps and not any(pj["gpus"] for pj in per_job):
        vllm = vllm_load()
        if vllm is not None:
            source = "vLLM /metrics (overlap unavailable)"

    return {
        "per_job": per_job,
        "node": node,
        "source": source,
        "jobs": node_jobs(),
        "nstate": nstate,
        "vllm": vllm,
        "gpu_group": our_group or "A",
        "env": env,
        "card_util": card_util,
    }


def bar(pct: float, kind: str, width: int = 8) -> str:
    """A fixed-width bar and its percentage.

    util — green when busy, yellow when partly used, red when idle.
    mem / load — occupancy: cyan→magenta→red as it fills toward saturation.
    """
    pct = max(0.0, min(100.0, pct))
    fill = round(pct / 100 * width)
    if kind == "util":
        colour = "green" if pct >= 70 else "yellow" if pct >= 30 else "red"
    else:  # mem, load — same fullness ramp
        colour = "red" if pct >= 90 else "magenta" if pct >= 70 else "cyan"
    bars = "█" * fill + "░" * (width - fill)
    return f"[{colour}]{bars}[/] {pct:3.0f}%"


# The queue is what a reader scrolls for, so it is given every line the panels
# above it do not need, and never fewer than this many rows while there are
# that many jobs to show.
MIN_JOB_ROWS = 10
# Lines the job table spends on its own frame: the header and the rule.
JOB_CHROME = 2


def _is_pending(state: str) -> bool:
    return state.strip().upper() in ("PENDING", "PD")


def _is_running(state: str) -> bool:
    return state.strip().upper().startswith("R")


def _job_group(acct: str, resv: str) -> str | None:
    return _acct_group(acct) or _resv_group(resv)


def ordered_jobs(jobs: list[list[str]]) -> list[list[str]]:
    """Running jobs first, then pending; within each, group A, B, then the rest."""

    def key(row: list[str]) -> tuple[int, int]:
        r = (list(row) + [""] * 9)[:9]
        group = _job_group(r[2], r[8]) or "Z"
        return (1 if _is_pending(r[5]) else 0, {"A": 0, "B": 1}.get(group, 2))

    return sorted(jobs, key=key)


def title_line(data: dict) -> Text:
    """The node in one line: its state, the group sampled and the cards held."""
    nstate = data.get("nstate", "unknown")
    state_word = nstate.split()[0] if nstate.strip() else "unknown"
    low = state_word.lower()
    # A drained or down node is the usual reason there are no card rows.
    if "drain" in low or "down" in low:
        nshow = f"[bold red]NODE {state_word.upper()}[/]"
    elif "alloc" in low or "mix" in low:
        nshow = f"[green]{state_word}[/]"
    else:
        nshow = f"[dim]{state_word}[/]"

    per_job = data.get("per_job")
    gpus = data.get("gpus") or []
    # The allocated count comes from the job table and needs no cgroup, so it
    # is authoritative. The card rows can only sample cards in a cgroup one of
    # this account's steps can join, so the sampled count can be lower even
    # when every card is busy; both are stated so neither is misread.
    alloc_gpu = 0
    for r in data.get("jobs", []):
        if len(r) > 7 and _is_running(r[5]):
            tail = r[7].strip().rsplit(":", 1)[-1]
            digits = "".join(c for c in tail if c.isdigit())
            if digits:
                alloc_gpu += int(digits)
    bits = [f"[bold]{NODE}[/]", time.strftime("%H:%M:%S"), nshow]
    if per_job or gpus or data.get("vllm") is not None:
        bits.append(f"[bold]Group {data.get('gpu_group', 'A')}[/]")
    else:
        bits.append("[dim]no group sampled[/]")
    if alloc_gpu:
        bits.append(f"[bold]{alloc_gpu}/{N_GPUS} GPUs alloc[/]")
    n_cards = (
        len(gpus)
        if gpus
        else (sum(len(pj.get("gpus", [])) for pj in per_job) if per_job else 0)
    )
    if n_cards:
        bits.append(f"{n_cards} sampled")
    line = "  ·  ".join(bits)
    if "drain" in low or "down" in low:
        line += (
            f"\n[red]NODE {nstate} — no jobs can run; needs admin: "
            f"scontrol update nodename={NODE} state=resume[/]"
        )
    return Text.from_markup(line)


def card_table(data: dict, window: int) -> Table:
    """One row per sampled card, one row for the cards that cannot be read.

    Memory is shown once: on a loaded engine the instantaneous and windowed
    figures are the same number, and where they differ the instantaneous one is
    the one that says whether a card is full now.
    """
    per_job = data.get("per_job")  # one-shot path (per-process)
    gpus = data.get("gpus", [])  # live path (by card)
    vllm = data.get("vllm")

    a = Table(box=box.SIMPLE_HEAVY, padding=(0, 1), expand=False, pad_edge=False)
    a.add_column("card", style="bold", justify="left", no_wrap=True, width=18)
    for h in ("util now", f"util {window}s", "mem"):
        a.add_column(h, justify="left", no_wrap=True, width=13)

    def vllm_row() -> None:
        kv = vllm.get("kv", 0.0) * 100.0
        a.add_row(
            "vllm kv-cache",
            bar(kv, "mem"),
            "",
            f"[dim]{vllm.get('running', 0.0):.0f} active · "
            f"{vllm.get('waiting', 0.0):.0f} queued[/]",
        )

    if per_job is not None:
        # A running job whose sampler returned nothing is shown unsampled,
        # never dropped, so every owned process stays visible.
        any_card = False
        for pj in per_job:
            nm = (pj.get("name") or pj.get("jobid") or "?")[:16]
            for phys, un, ua, mn, _ma in pj["gpus"]:
                a.add_row(
                    f"{nm} gpu{phys}", bar(un, "util"), bar(ua, "util"), bar(mn, "mem")
                )
                any_card = True
            if not pj["gpus"]:
                a.add_row(nm, "[yellow]unsampled[/]", "", "[dim]no reading[/]")
        if not any_card and vllm is not None:
            vllm_row()
        elif not per_job:
            a.add_row("—", "[dim]—[/]", "", "")
    elif gpus:
        smp = (
            gpus if isinstance(gpus, dict) else {str(i): v for i, v in enumerate(gpus)}
        )
        alloc_idx = data.get("alloc_idx") or set()
        sampled = sorted(int(k) for k in smp if str(k).isdigit())
        for idx in sampled:
            un, ua, mn, _ma = smp[str(idx)]
            a.add_row(f"gpu {idx}", bar(un, "util"), bar(ua, "util"), bar(mn, "mem"))
        # Cards a running job holds in a cgroup none of this account's steps
        # can join: busy, and unreadable. They share one row, because each
        # would say the same thing.
        hidden = sorted(alloc_idx - set(sampled))
        if hidden:
            a.add_row(
                "gpu " + " ".join(str(i) for i in hidden),
                "[yellow]unsampled[/]",
                "",
                "[dim]allocated[/]",
            )
    elif vllm is not None:
        vllm_row()
    else:
        a.add_row("—", "[dim]—[/]", "", "")

    node = data.get("node")
    if node:
        cpu, used, total = node
        ram = 100 * used / max(total, 1)
        nm = data.get("node_mean")  # (cpu_mean, ram_mean) in --live; else None
        a.add_row(
            "[dim]host[/]",
            "cpu " + bar(cpu, "load", 4),
            "cpu " + bar(nm[0], "load", 4) if nm else "",
            "ram " + bar(ram, "load", 4),
        )
    return a


def job_table(jobs: list[list[str]], rows: int | None = None) -> Table:
    """The node's queue, any user: running jobs, then pending ones.

    This is where another group's load is visible, since its cards cannot be
    sampled. ``rows`` bounds the table to the lines the screen has left; the
    jobs that do not fit are counted on a last line rather than cut silently.
    """
    b = Table(
        box=box.SIMPLE, padding=(0, 1), expand=False, pad_edge=False, show_edge=False
    )
    for c, kw in [
        ("jobid", {}),
        ("user", {}),
        ("acct", {}),
        ("name", {"max_width": 32}),
        ("state", {}),
        ("elapsed", {}),
        ("gres", {}),
    ]:
        b.add_column(c, style="dim", no_wrap=True, **kw)

    if not jobs:
        b.add_row("[dim]node idle[/]", "", "", "", "", "", "")
        return b
    ordered = ordered_jobs(jobs)
    shown = ordered
    if rows is not None and len(ordered) > rows:
        shown = ordered[: max(1, rows - 1)]
    for row in shown:
        jid, user, acct, name, _part, state, elapsed, gres, resv = (
            list(row) + [""] * 9
        )[:9]
        pending = _is_pending(state)
        colour = (
            "dim"
            if pending
            else {"A": "green", "B": "cyan"}.get(_job_group(acct, resv) or "", "dim")
        )
        cells = (jid, user, acct, name, state, elapsed, gres.replace("gres/", ""))
        b.add_row(*[f"[{colour}]{v}[/]" for v in cells])
    rest = ordered[len(shown) :]
    if rest:
        waiting = sum(1 for r in rest if _is_pending((list(r) + [""] * 6)[5]))
        b.add_row(
            f"[dim]+{len(rest)} more[/]",
            "",
            "",
            "",
            f"[dim]{waiting} pending[/]",
            "",
            "",
        )
    return b


def _line_count(parts: list, width: int) -> int:
    """Lines ``parts`` occupy when drawn ``width`` columns wide."""
    import io

    probe = Console(
        width=width, file=io.StringIO(), force_terminal=False, color_system=None
    )
    return len(probe.render_lines(Group(*parts), probe.options, pad=False))


def render(data: dict, window: int, size: tuple[int, int] | None = None) -> Group:
    """The whole view: cards, the served lane, then the queue.

    ``size`` is the screen as ``(columns, lines)``. With it, the queue is given
    every line the panels above leave, and when that is fewer than
    ``MIN_JOB_ROWS`` the serve grid closes the blank lines between its bands to
    make room. Without it (a one-shot reading) nothing is bounded.
    """
    jobs = data.get("jobs") or []
    model = (_serve_cache.get("model") or ("", ""))[0]
    serve_idx = serve_gpu_idx(jobs, model)
    serve_state()
    live_doc = _watch_live()
    gpu_count = _serve_cache.get("cards") or 0

    def top(tight: bool) -> list:
        return [
            title_line(data),
            card_table(data, window),
            serve_table(live_doc, gpu_count, data.get("env"), serve_idx, tight),
        ]

    parts = top(False)
    rows = None
    if size is not None:
        width, height = size
        wanted = min(MIN_JOB_ROWS, max(1, len(jobs)))
        rows = height - _line_count(parts, width) - JOB_CHROME
        if rows < wanted:
            parts = top(True)
            rows = height - _line_count(parts, width) - JOB_CHROME
        rows = max(rows, 1)
    return Group(*parts, job_table(jobs, rows))


def run_live(window: int) -> int:
    """Live view backed by ONE long-lived overlap step per running job.

    A background reader thread consumes each sampler's streaming output and
    keeps only the latest per-card + host values; the main loop redraws at 1 Hz
    and maintains the rolling `window`-second means itself. This costs one SLURM
    step per session — not one per second — so it can never exhaust a job's
    lifetime step counter (a step per second spends about 40000 a day and locks
    --overlap out for the rest of the served job's life).

    The set of running jobs is re-checked every 10 s: new jobs get a new sampler
    spawned immediately, and a sampler whose process exits removes its GPU UUIDs
    from the live dict so stale bars collapse right away.
    """
    import threading
    from collections import deque

    console = Console()

    latest: dict = {"gpus": {}, "node": None, "env": {}}
    lock = threading.Lock()

    # Track active sampler procs by job ID so we can detect new/gone jobs.
    active_procs: dict[str, subprocess.Popen] = {}
    procs_lock = threading.Lock()

    def reader(job_id: str, proc: subprocess.Popen) -> None:
        """Consume one sampler's stream; clean up its GPU UUIDs on exit."""
        owned_uuids: set[str] = set()
        for line in proc.stdout:  # type: ignore[union-attr]
            p = line.rstrip("\n").split("\t")
            with lock:
                if p[0] == "GPU" and len(p) >= 5:
                    # GPU, phys, uuid, util_now, mem_now: keyed by physical index
                    latest["gpus"][p[1]] = (float(p[3]), float(p[4]))
                    if len(p) >= 8:
                        latest["env"][p[1]] = _env_fields(p[5], p[6], p[7])
                    owned_uuids.add(p[1])
                elif p[0] == "NODE" and len(p) >= 4:
                    latest["node"] = (float(p[1]), float(p[2]), float(p[3]))
        # Sampler exited — remove stale GPU readings so bars collapse immediately.
        with lock:
            for u in owned_uuids:
                latest["gpus"].pop(u, None)
                latest["env"].pop(u, None)
        with procs_lock:
            active_procs.pop(job_id, None)

    def launch_sampler(job_id: str) -> None:
        proc = subprocess.Popen(
            [
                "srun",
                "--jobid",
                job_id,
                "--overlap",
                "python3",
                "-u",
                SAMPLER,
                "stream",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        with procs_lock:
            active_procs[job_id] = proc
        threading.Thread(target=reader, args=(job_id, proc), daemon=True).start()

    # Initial launch
    steps, our_group = overlap_steps()
    gpu_group = our_group or "A"
    # (Re)attach a sampler to every owned job that has no live one — at startup,
    # when a new job appears, AND when a previously-launched sampler died (a
    # transient `srun --overlap` failure or a dropped step).  Without the
    # death-recovery a job whose sampler ever failed had its GPUs vanish from the
    # view for the whole session — which made the node look like it had fewer
    # cards than it does (e.g. a 6-card training job shown while the co-resident
    # 2-card serve's cards were missing).  A per-job cooldown bounds step-quota
    # (MaxStepCount) use: a hard-failing job is retried at most every
    # relaunch_cooldown seconds, not every tick.
    relaunch_cooldown = 20.0
    last_launch: dict[str, float] = {}

    def ensure_sampler(job_id: str) -> None:
        with procs_lock:
            if job_id in active_procs:
                return  # a live sampler is already streaming
        if time.monotonic() - last_launch.get(job_id, -1e9) < relaunch_cooldown:
            return  # cooled down — don't burn step quota
        last_launch[job_id] = time.monotonic()
        launch_sampler(job_id)

    for j in steps:
        ensure_sampler(j)

    # Per-UUID rolling-mean buffers — keyed by UUID so card count changes are safe.
    gpu_buf: dict[str, deque] = {}
    cpu_buf: deque = deque(maxlen=max(1, window))
    ram_buf: deque = deque(maxlen=max(1, window))

    def reap() -> None:
        """Stop every sampler this session started.

        Without this the srun steps outlive the watcher: measured on this host,
        two of them survived their parent by nearly three hours, reparented to
        init, still running nvidia-smi once a second and each still holding a
        SLURM step against a job that had no idea they were there. A step is a
        bounded resource, and one leaked per job per session exhausts it.
        """
        with procs_lock:
            doomed = list(active_procs.values())
            active_procs.clear()
        for proc in doomed:
            with contextlib.suppress(Exception):
                proc.terminate()
        for proc in doomed:
            try:
                proc.wait(timeout=3)
            except Exception:
                with contextlib.suppress(Exception):
                    proc.kill()

    # A terminal signal has to run the cleanup too, and Python does not unwind
    # for one by default -- `timeout`, a window close and a plain `kill` all
    # arrive this way, which is exactly how the orphans above were made.
    import signal

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    # screen=True uses the terminal's alternate buffer (like `watch`): a clean
    # full-screen view redrawn in place, restored on exit.
    try:
        with Live(console=console, auto_refresh=False, screen=True) as live:
            live.update(console.render_str("[dim]sampling…[/]"), refresh=True)
            while True:
                t0 = time.monotonic()

                # Re-check running jobs every tick — one extra squeue call, no
                # more expensive than the sinfo + 2×squeue already called each
                # second for node_state() + node_jobs().  Only call srun --overlap
                # (launch_sampler) for brand-new job IDs; never retry a known job
                # even if its sampler exited, to avoid burning the step quota.
                current_steps, current_group = overlap_steps()
                current_ids = set(current_steps)
                if current_group:
                    gpu_group = current_group
                for j in list(last_launch):  # forget cooldown for jobs that ended
                    if j not in current_ids:
                        last_launch.pop(j, None)
                for j in current_steps:
                    ensure_sampler(j)  # (re)attach if missing or sampler died

                with lock:
                    gpu_now = dict(latest["gpus"])
                    node_now = latest["node"]
                    env_now = dict(latest["env"])
                card_util = {k: v[0] for k, v in gpu_now.items()}

                data: dict = {
                    "nstate": node_state(),
                    "jobs": node_jobs(),
                    "gpu_group": gpu_group,
                    "vllm": None,
                    "source": "",
                    "gpus": [],
                    "node": node_now,
                    "env": env_now,
                    "card_util": card_util,
                }
                run_ids = [
                    r[0]
                    for r in data["jobs"]
                    if len(r) > 5 and r[5].strip().upper().startswith("R")
                ]
                data["alloc_idx"] = allocated_gpu_idx(run_ids)

                if gpu_now:
                    keys = sorted(gpu_now, key=lambda k: int(k) if k.isdigit() else 999)
                    # Drop buffers for GPUs that have gone away.
                    for u in list(gpu_buf):
                        if u not in gpu_now:
                            del gpu_buf[u]
                    # Append the sample; rolling means are per physical card.
                    gpu_map: dict = {}
                    for u in keys:
                        if u not in gpu_buf:
                            gpu_buf[u] = deque(maxlen=max(1, window))
                        gpu_buf[u].append(gpu_now[u])
                        samples = gpu_buf[u]
                        util_now, mem_now = gpu_now[u]
                        util_mean = sum(s[0] for s in samples) / len(samples)
                        mem_mean = sum(s[1] for s in samples) / len(samples)
                        gpu_map[u] = (util_now, util_mean, mem_now, mem_mean)
                    data["gpus"] = gpu_map
                else:
                    # No per-GPU stream (no overlappable step, or step exhausted):
                    # fall back to the vLLM /metrics activity proxy.
                    data["vllm"] = vllm_load()

                if node_now:
                    cpu, used, total = node_now
                    cpu_buf.append(cpu)
                    ram_buf.append(100.0 * used / max(total, 1))
                    data["node_mean"] = (
                        sum(cpu_buf) / len(cpu_buf),
                        sum(ram_buf) / len(ram_buf),
                    )

                live.update(render(data, window, console.size), refresh=True)
                elapsed = time.monotonic() - t0
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)

    finally:
        reap()


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description="windowed GPU watcher")
    ap.add_argument(
        "window",
        nargs="?",
        type=int,
        default=30,
        help="averaging window in seconds (default 30)",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="refresh forever at 1 Hz; the mean columns are a "
        "rolling mean over the last `window` seconds, fed by a "
        "single long-lived on-node sampler",
    )
    args = ap.parse_args()

    if args.live:
        return run_live(args.window)

    # Resolve the serve's own card count and context window before rendering
    # once, so the header and the environment band do not read as empty on the
    # first frame. The sample() call below already spends the window on the
    # cards; this costs one extra scrape.
    serve_state()
    prime_callers()
    data = sample(args.window)
    serve_state()
    Console().print(render(data, args.window))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
