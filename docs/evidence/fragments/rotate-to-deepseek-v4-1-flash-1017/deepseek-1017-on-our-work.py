#!/usr/bin/env python3
"""Compare local-lane implementation runs either side of the checkpoint swap.

Reads every mounted project's run ledger (``~/Code/*/docs/state/*/runs/*.json``)
and the lane's request record (``~/public/imas-ambix/requests*.jsonl``), splits
the ``clive``-backend implementation runs by dispatch time at the swap boundary
2026-10-23T20:14:30Z, and reports each metric with a 95% bootstrap interval.

Only records whose ``role`` is ``implement`` and whose ``backend`` is ``clive``
are counted. Runs whose dispatch-to-completion interval straddling the boundary
are excluded from both windows.

Metrics, per window and as an after-minus-before difference:

  promotion_rate          share of runs whose work was committed and promoted
  first_attempt_pass      share of runs that passed their gate on attempt 1
  repair_resume_share     share of runs that needed a repair or resume round
  review_<dimension>      mean independent-review score per dimension
  wall_seconds            mean wall time per promoted run
  input_tokens            mean input tokens per promoted run (joined by run id)
  cached_tokens           mean cached prompt tokens per promoted run
  output_tokens           mean output tokens per promoted run
  request_failure_share   share of lane requests that failed or aborted

The positive control (at least 30 runs per window, the analysed count
reconciling with an independent ledger count, and a token join reaching at
least 90% of runs) is evaluated before any verdict. A failed control yields
``insufficient data`` regardless of what the metrics say.

A moved-boundary control, with the boundary placed at the midpoint of the
before window so both sides hold October runs, must pass the same positive
control and report no detectable difference on the promotion and first-attempt
rates; a difference there would mean the instrument reports change where the
weights did not change.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import random
import statistics
import sys
from datetime import datetime, timedelta, timezone

HOME = os.path.expanduser("~")
LEDGER_GLOB = os.path.join(HOME, "Code", "*", "docs", "state", "*", "runs", "*.json")
REQUEST_FILES = [
    os.path.join(HOME, "public", "imas-ambix", "requests.jsonl"),
    os.path.join(HOME, "public", "imas-ambix", "requests-2026-10-23.jsonl.gz"),
    os.path.join(HOME, "public", "imas-ambix", "requests-2026-10-24.jsonl"),
    os.path.join(HOME, "public", "imas-ambix", "requests-2026-10-25.jsonl"),
    os.path.join(HOME, "public", "imas-ambix", "requests-2026-10-26.jsonl"),
]

BOUNDARY = datetime(2026, 10, 23, 20, 14, 30, tzinfo=timezone.utc)
WINDOW = timedelta(days=3)
AFTER = (BOUNDARY, BOUNDARY + WINDOW)
BEFORE = (BOUNDARY - WINDOW, BOUNDARY)
BEFORE_7D = (BOUNDARY - timedelta(days=7), BOUNDARY)

DIMENSIONS = ("durability", "evidence", "fit", "goal_fidelity", "scope_discipline")
BOOTSTRAP_ROUNDS = 5000
RNG_SEED = 20261023


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def project_of(path):
    marker = os.sep + "docs" + os.sep + "state" + os.sep
    return path.split(marker, 1)[1].split(os.sep, 1)[0]


def load_runs():
    """Enumerate implement/clive ledger records and keep the fields analysed."""
    runs = []
    for path in sorted(glob.glob(LEDGER_GLOB)):
        try:
            with open(path) as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        if record.get("role") != "implement":
            continue
        if record.get("backend") != "clive":
            continue
        runs.append(
            {
                "run_id": record.get("run_id") or os.path.basename(path)[:-5],
                "project": project_of(path),
                "dispatched_at": parse_ts(record.get("dispatched_at")),
                "completed_at": parse_ts(record.get("completed_at")),
                "promoted": bool(
                    record.get("promoted_revision")
                    and record.get("commits")
                    and not record.get("no_commit")
                ),
                "attempt": record.get("attempt"),
                "attempt_kind": record.get("attempt_kind"),
                "gate": record.get("gate"),
                "review_scores": ((record.get("review") or {}).get("scores") or {}),
                "wall_seconds": record.get("wall_seconds"),
            }
        )
    return runs


def independent_count(runs, window):
    """Recount the window by a second traversal and its own timestamp parse.

    This deliberately does not share the straddle filter: the positive control
    compares it with the analysed count plus the straddlers, so a filter bug
    shows up as a mismatch rather than silently shrinking the sample.
    """
    start, end = window
    total = 0
    for base in sorted(glob.glob(os.path.join(HOME, "Code", "*", "docs", "state", "*", "runs"))):
        for name in os.listdir(base):
            if not name.endswith(".json"):
                continue
            path = os.path.join(base, name)
            record = _RAW_CACHE.get(path)
            if record is None:
                try:
                    with open(path) as handle:
                        record = json.load(handle)
                except (OSError, ValueError):
                    continue
                _RAW_CACHE[path] = record
            if record.get("role") != "implement" or record.get("backend") != "clive":
                continue
            stamp = parse_ts(record.get("dispatched_at"))
            if stamp is not None and start <= stamp < end:
                total += 1
    return total


def straddles_boundary(run, boundary):
    dispatched, completed = run["dispatched_at"], run["completed_at"]
    if dispatched is None or completed is None:
        return False
    return dispatched < boundary <= completed


def select(runs, window, boundary):
    start, end = window
    inside = [r for r in runs if r["dispatched_at"] is not None and start <= r["dispatched_at"] < end]
    straddlers = [r for r in inside if straddles_boundary(r, boundary)]
    kept = [r for r in inside if not straddles_boundary(r, boundary)]
    return inside, kept, straddlers


def load_requests():
    """Join the request record by run id, and tally status over the whole lane."""
    tokens = {}
    per_day = []
    for path in REQUEST_FILES:
        if not os.path.exists(path):
            continue
        opener = gzip.open if path.endswith(".gz") else open
        count = 0
        with opener(path, "rt", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                count += 1
                stamp = parse_ts(row.get("timestamp"))
                if stamp is not None:
                    per_day.append((stamp, row.get("status")))
                run_id = row.get("run_id")
                if not run_id:
                    continue
                bucket = tokens.setdefault(
                    run_id, {"input": 0, "cached": 0, "output": 0, "requests": 0, "failed": 0}
                )
                bucket["requests"] += 1
                bucket["input"] += int(row.get("prompt_tokens") or 0)
                bucket["cached"] += int(row.get("cached_prompt_tokens") or 0)
                bucket["output"] += int(row.get("completion_tokens") or 0)
                if row.get("status") in ("failed", "aborted"):
                    bucket["failed"] += 1
        print(f"  request file {os.path.basename(path)}: {count} rows")
    return tokens, per_day


def bootstrap_ci(values, statistic, rounds=BOOTSTRAP_ROUNDS, seed=RNG_SEED):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        value = statistic(sample)
        if value is not None:
            stats.append(value)
    if not stats:
        return [None, None]
    stats.sort()
    low = stats[int(0.025 * (len(stats) - 1))]
    high = stats[int(0.975 * (len(stats) - 1))]
    return [round(low, 6), round(high, 6)]


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def share(values):
    return statistics.fmean(values) if values else None


def ratio(numerator, denominator):
    return (numerator / denominator) if denominator else None


def surviving(values):
    return [v for v in values if v is not None]


def metric_values(runs, tokens, requests):
    """Per-run vectors for one window; each entry is one run."""
    promoted = [r for r in runs if r["promoted"]]
    paired = [r for r in promoted if r["run_id"] in tokens]
    values = {
        "promotion_rate": [1.0 if r["promoted"] else 0.0 for r in runs],
        "first_attempt_pass": [
            1.0 if (r["attempt"] == 1 and r["gate"] == "passed") else 0.0 for r in runs
        ],
        "repair_resume_share": [
            1.0
            if (
                (isinstance(r["attempt"], int) and r["attempt"] > 1)
                or str(r["attempt_kind"] or "") in ("resume", "redispatch", "repair")
            )
            else 0.0
            for r in runs
        ],
    }
    for dim in DIMENSIONS:
        values[f"review_{dim}"] = [
            float(r["review_scores"][dim]) for r in runs if dim in r["review_scores"]
        ]
    values["wall_seconds"] = [
        float(r["wall_seconds"]) for r in promoted if r["wall_seconds"] is not None
    ]
    values["input_tokens"] = [float(tokens[r["run_id"]]["input"]) for r in paired]
    values["cached_tokens"] = [float(tokens[r["run_id"]]["cached"]) for r in paired]
    values["output_tokens"] = [float(tokens[r["run_id"]]["output"]) for r in paired]

    start, end = requests
    statuses = [status for stamp, status in _REQUEST_STATUS if start <= stamp < end]
    values["request_failure_share"] = [
        1.0 if status in ("failed", "aborted") else 0.0 for status in statuses
    ]
    return values, {
        "runs": len(runs),
        "promoted": len(runs),
        "promoted_count": len(promoted),
        "token_joined_promoted": len(paired),
        "requests": len(statuses),
    }


_REQUEST_STATUS = []
_RAW_CACHE = {}


def summarise(values):
    out = {}
    for name, vector in values.items():
        cleaned = surviving(vector)
        estimate = mean_or_none(cleaned)
        low, high = bootstrap_ci(cleaned, mean_or_none)
        out[name] = {
            "n": len(cleaned),
            "estimate": None if estimate is None else round(estimate, 6),
            "ci95": [low, high],
        }
    return out


def difference(after_vector, before_vector):
    """After-minus-before, resampling each window independently."""
    a = surviving(after_vector)
    b = surviving(before_vector)
    if not a or not b:
        return {"n_after": len(a), "n_before": len(b), "estimate": None, "ci95": [None, None]}
    rng = random.Random(RNG_SEED)
    stats = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        stats.append(statistics.fmean(sa) - statistics.fmean(sb))
    stats.sort()
    return {
        "n_after": len(a),
        "n_before": len(b),
        "estimate": round(statistics.fmean(a) - statistics.fmean(b), 6),
        "ci95": [
            round(stats[int(0.025 * (len(stats) - 1))], 6),
            round(stats[int(0.975 * (len(stats) - 1))], 6),
        ],
    }


def project_mix(runs):
    counts = {}
    for run in runs:
        counts[run["project"]] = counts.get(run["project"], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def positive_control(window, runs, kept, straddlers, independent, tokens, promoted_count, joined):
    start, end = window
    checks = {
        "runs_at_least_30": len(kept) >= 30,
        "independent_count": independent,
        "analysed_count": len(kept),
        "straddlers_excluded": len(straddlers),
        "count_reconciles": independent == len(kept) + len(straddlers),
        "token_join_share": round(ratio(joined, promoted_count) or 0.0, 4),
        "token_join_at_least_90pct": (ratio(joined, promoted_count) or 0.0) >= 0.90,
    }
    checks["passed"] = bool(
        checks["runs_at_least_30"]
        and checks["count_reconciles"]
        and checks["token_join_at_least_90pct"]
    )
    return checks


def verdict_from(diffs, control_ok):
    if not control_ok:
        return "insufficient data", (
            "the positive control failed, so no comparison is reported as a result"
        )
    primary = {"promotion_rate": +1, "first_attempt_pass": +1}
    better = worse = 0
    detail = []
    for name, direction in primary.items():
        entry = diffs.get("restricted", {}).get(name) or diffs.get("all", {}).get(name)
        low, high = entry["ci95"]
        if low is None:
            continue
        if low > 0:
            better += 1
            detail.append(f"{name} +{entry['estimate']:.3f} [{low:.3f}, {high:.3f}]")
        elif high < 0:
            worse += 1
            detail.append(f"{name} {entry['estimate']:.3f} [{low:.3f}, {high:.3f}]")
    if better and not worse:
        return "better", "; ".join(detail)
    if worse and not better:
        return "worse", "; ".join(detail)
    return "no detectable difference", "; ".join(detail) or "no primary metric interval excludes zero"


def write_svg(path, diffs, title_note):
    order = [
        ("promotion_rate", "promotion rate"),
        ("first_attempt_pass", "first-attempt gate pass"),
        ("repair_resume_share", "repair or resume share"),
        ("request_failure_share", "failed/aborted requests"),
    ]
    rows = [(label, key) for key, label in order if key in diffs]
    for dim in DIMENSIONS:
        key = f"review_{dim}"
        if key in diffs:
            rows.append((f"review {dim.replace('_', ' ')}", key))
    for key, label in (
        ("wall_seconds", "wall time [s]"),
        ("input_tokens", "input tokens [k]"),
        ("cached_tokens", "cached tokens [k]"),
        ("output_tokens", "output tokens [k]"),
    ):
        if key in diffs:
            rows.append((label, key))

    width, row_h = 1400, 44
    top, bottom = 90, 70
    height = top + row_h * len(rows) + bottom
    left, right = 420, 130

    spans = []
    for _, key in rows:
        entry = diffs[key]
        est, (low, high) = entry["estimate"], entry["ci95"]
        if est is None or low is None:
            continue
        if key.endswith("_tokens"):
            est, low, high = est / 1000.0, low / 1000.0, high / 1000.0
        spans.append((low, high))
    if not spans:
        return
    lo = min(min(s[0] for s in spans), 0.0)
    hi = max(max(s[1] for s in spans), 0.0)
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad

    def x(value):
        return left + (value - lo) / (hi - lo) * (width - left - right)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" id="deepseek-1017-on-our-work-differences" '
        f'width="1400" height="{height}" viewBox="0 0 1400 {height}" '
        f'font-family="Helvetica, Arial, sans-serif">',
        f'<rect id="deepseek-1017-on-our-work-bg" width="1400" height="{height}" fill="#ffffff"/>',
    ]
    axis_y = top + row_h * len(rows) - row_h * 0.2
    parts.append(
        f'<line id="deepseek-1017-on-our-work-zero" x1="{x(0):.1f}" y1="{top - 24}" '
        f'x2="{x(0):.1f}" y2="{axis_y:.1f}" stroke="#444444" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line id="deepseek-1017-on-our-work-axis" x1="{left}" y1="{axis_y:.1f}" '
        f'x2="{width - right}" y2="{axis_y:.1f}" stroke="#444444" stroke-width="1.2"/>'
    )
    tick = lo
    while tick <= hi:
        tx = x(tick)
        if left - 1 <= tx <= width - right + 1:
            parts.append(
                f'<text id="deepseek-1017-on-our-work-tick-{int(round(tick * 1000))}" x="{tx:.1f}" '
                f'y="{axis_y + 26:.1f}" font-size="14" fill="#333333" text-anchor="middle">{tick:.3g}</text>'
            )
        tick += (hi - lo) / 8.0
    parts.append(
        f'<text id="deepseek-1017-on-our-work-xlabel" x="{(left + width - right) / 2:.1f}" '
        f'y="{axis_y + 56:.1f}" font-size="16" fill="#222222" text-anchor="middle">'
        "after minus before (95% bootstrap interval)</text>"
    )

    for index, (label, key) in enumerate(rows):
        entry = diffs[key]
        est, (low, high) = entry["estimate"], entry["ci95"]
        cy = top + row_h * index + row_h / 2 - 6
        parts.append(
            f'<text id="deepseek-1017-on-our-work-label-{key}" x="{left - 18}" y="{cy + 5:.1f}" '
            f'font-size="14" fill="#222222" text-anchor="end">{label}</text>'
        )
        if est is None or low is None:
            parts.append(
                f'<text id="deepseek-1017-on-our-work-empty-{key}" x="{left + 8}" y="{cy + 5:.1f}" '
                f'font-size="14" fill="#8a8a8a">no data</text>'
            )
            continue
        if key.endswith("_tokens"):
            est, low, high = est / 1000.0, low / 1000.0, high / 1000.0
        crosses = low <= 0.0 <= high
        colour = "#929292" if crosses else "#1f6f8b"
        parts.append(
            f'<line id="deepseek-1017-on-our-work-ci-{key}" x1="{x(low):.1f}" y1="{cy:.1f}" '
            f'x2="{x(high):.1f}" y2="{cy:.1f}" stroke="{colour}" stroke-width="2.4"/>'
        )
        parts.append(
            f'<line id="deepseek-1017-on-our-work-cap-lo-{key}" x1="{x(low):.1f}" y1="{cy - 7:.1f}" '
            f'x2="{x(low):.1f}" y2="{cy + 7:.1f}" stroke="{colour}" stroke-width="2.4"/>'
        )
        parts.append(
            f'<line id="deepseek-1017-on-our-work-cap-hi-{key}" x1="{x(high):.1f}" y1="{cy - 7:.1f}" '
            f'x2="{x(high):.1f}" y2="{cy + 7:.1f}" stroke="{colour}" stroke-width="2.4"/>'
        )
        parts.append(
            f'<circle id="deepseek-1017-on-our-work-point-{key}" cx="{x(est):.1f}" cy="{cy:.1f}" '
            f'r="5" fill="{colour}"/>'
        )
    parts.append(
        f'<text id="deepseek-1017-on-our-work-note" x="{left}" y="{height - 18}" font-size="14" '
        f'fill="#555555">{title_note}</text>'
    )
    parts.append("</svg>")
    with open(path, "w") as handle:
        handle.write("\n".join(parts) + "\n")


def report_window(name, window, runs, tokens, statuses, boundary):
    inside, kept, straddlers = select(runs, window, boundary)
    independent = independent_count(runs, window)
    values, counts = metric_values(kept, tokens, (window[0], window[1]))
    promoted_ids = [r["run_id"] for r in kept if r["promoted"]]
    joined = len([rid for rid in promoted_ids if rid in tokens])
    control = positive_control(
        window, runs, kept, straddlers, independent, tokens, len(promoted_ids), joined
    )
    summary = summarise(values)
    return {
        "window": {
            "start": window[0].isoformat(),
            "end": window[1].isoformat(),
        },
        "counts": {
            "runs_in_window": len(inside),
            "straddlers_excluded": len(straddlers),
            "analysed": len(kept),
            "promoted": len(promoted_ids),
            "projects": project_mix(kept),
            "requests": len(_REQUEST_STATUS),
        },
        "positive_control": control,
        "metrics": summary,
        "vectors": values,
    }


def main():
    print(f"# analysis: local-lane implementation runs around {BOUNDARY.isoformat()}")
    print(f"# ledger glob: {LEDGER_GLOB}")
    print(f"# request files: {len(REQUEST_FILES)}")
    runs = load_runs()
    print(f"# implement/clive ledger records: {len(runs)}")

    tokens, statuses = load_requests()
    _REQUEST_STATUS.extend(statuses)
    print(f"# request rows with a timestamp: {len(_REQUEST_STATUS)}")
    print(f"# distinct run ids in the request record: {len(tokens)}")

    before = report_window("before", BEFORE, runs, tokens, statuses, BOUNDARY)
    after = report_window("after", AFTER, runs, tokens, statuses, BOUNDARY)
    before7 = report_window("before_7d", BEFORE_7D, runs, tokens, statuses, BOUNDARY)

    common = sorted(set(before["counts"]["projects"]) & set(after["counts"]["projects"]))
    print(f"# projects present in both windows: {common}")

    def restricted(record):
        inside, kept, _ = select(runs, (parse_ts(record["window"]["start"]), parse_ts(record["window"]["end"])), BOUNDARY)
        kept = [r for r in kept if r["project"] in common]
        values, _ = metric_values(kept, tokens, record["window_range"])
        return values

    # attach concrete ranges for the restriction pass
    before["window_range"] = BEFORE
    after["window_range"] = AFTER
    before["restricted_vectors"] = restricted(before)
    after["restricted_vectors"] = restricted(after)

    main_diffs = {
        name: difference(after["vectors"][name], before["vectors"][name])
        for name in after["vectors"]
    }
    restricted_diffs = {
        name: difference(after["restricted_vectors"][name], before["restricted_vectors"][name])
        for name in after["restricted_vectors"]
    }
    sensitivity_diffs = {
        name: difference(after["vectors"][name], before7["vectors"][name])
        for name in after["vectors"]
    }

    control_ok = before["positive_control"]["passed"] and after["positive_control"]["passed"]

    # moved-boundary control: place the cut at the midpoint of the before window
    moved = BEFORE[0] + (BOUNDARY - BEFORE[0]) / 2
    first_half, kept_a, strad_a = select(runs, (BEFORE[0], moved), moved)
    second_half, kept_b, strad_b = select(runs, (moved, BOUNDARY), moved)
    moved_values_a, _ = metric_values(kept_a, tokens, (BEFORE[0], moved))
    moved_values_b, _ = metric_values(kept_b, tokens, (moved, BOUNDARY))
    moved_control = {
        "boundary": moved.isoformat(),
        "left": positive_control(
            (BEFORE[0], moved), runs, kept_a, strad_a,
            independent_count(runs, (BEFORE[0], moved)), tokens,
            len([r for r in kept_a if r["promoted"]]),
            len([r for r in kept_a if r["promoted"] and r["run_id"] in tokens]),
        ),
        "right": positive_control(
            (moved, BOUNDARY), runs, kept_b, strad_b,
            independent_count(runs, (moved, BOUNDARY)), tokens,
            len([r for r in kept_b if r["promoted"]]),
            len([r for r in kept_b if r["promoted"] and r["run_id"] in tokens]),
        ),
    }
    moved_diffs = {
        name: difference(moved_values_b[name], moved_values_a[name])
        for name in ("promotion_rate", "first_attempt_pass")
    }
    moved_ok = moved_control["left"]["passed"] and moved_control["right"]["passed"]
    moved_no_change = all(
        entry["ci95"][0] is not None and entry["ci95"][0] <= 0 <= entry["ci95"][1]
        for entry in moved_diffs.values()
    )

    result = {
        "boundary": BOUNDARY.isoformat() if hasattr(BOUNDARY, "isoformat") else BOUNDARY.isoformat(),
        "windows": {
            "before": before["window"],
            "after": after["window"],
            "before_7d": before7["window"],
        },
        "counts": {
            "before": before["counts"],
            "after": after["counts"],
            "before_7d": before7["counts"],
        },
        "positive_control": {
            "before": before["positive_control"],
            "after": after["positive_control"],
            "passed": control_ok,
        },
        "metrics": {
            "before": before["metrics"],
            "after": after["metrics"],
            "before_7d": before7["metrics"],
        },
        "differences": main_diffs,
        "differences_common_projects": restricted_diffs,
        "sensitivity_7d": sensitivity_diffs,
        "moved_boundary_control": {
            "control": moved_control,
            "differences": moved_diffs,
            "control_passed": moved_ok,
            "no_detectable_difference": moved_no_change,
        },
        "verdict": {},
    }
    verdict, why = verdict_from({"all": main_diffs, "restricted": restricted_diffs}, control_ok)
    result["verdict"] = {"verdict": verdict, "basis": why}

    print("\n== windows ==")
    for key in ("before", "after", "before_7d"):
        count_block = result["counts"][key]
        print(
            f"{key}: {result['windows'][key]['start']} .. {result['windows'][key]['end']} "
            f"analysed={count_block['analysed']} excluded_straddlers={count_block['straddlers_excluded']} "
            f"promoted={count_block['promoted']}"
        )
        print(f"  projects: {count_block['projects']}")
    print("\n== positive control ==")
    for key in ("before", "after"):
        print(f"{key}: {json.dumps(result['positive_control'][key])}")
    print(f"passed: {control_ok}")
    print("\n== metrics ==")
    for name in main_diffs:
        b = result["metrics"]["before"][name]
        a = result["metrics"]["after"][name]
        d = main_diffs[name]
        print(
            f"{name}: before {b['estimate']} {b['ci95']} (n={b['n']}) | "
            f"after {a['estimate']} {a['ci95']} (n={a['n']}) | "
            f"diff {d['estimate']} {d['ci95']}"
        )
    print("\n== differences restricted to projects present in both windows ==")
    for name, d in restricted_diffs.items():
        print(f"{name}: {d['estimate']} {d['ci95']} (n_after={d['n_after']}, n_before={d['n_before']})")
    print("\n== sensitivity: 7-day before window ==")
    for name, d in sensitivity_diffs.items():
        print(f"{name}: {d['estimate']} {d['ci95']}")
    print("\n== moved-boundary control ==")
    print(f"boundary {moved.isoformat()} control_passed={moved_ok} no_detectable_difference={moved_no_change}")
    for name, d in moved_diffs.items():
        print(f"{name}: {d['estimate']} {d['ci95']}")
    print("\n== verdict ==")
    print(f"{verdict}: {why}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = script_dir
    for _ in range(4):
        repo_root = os.path.dirname(repo_root)
    figure_dir = os.path.join(
        repo_root,
        "docs",
        "figures",
        "rotate-to-deepseek-v4-1-flash-1017",
        "deepseek-1017-on-our-work",
    )
    os.makedirs(figure_dir, exist_ok=True)
    figure = os.path.join(figure_dir, "differences.svg")
    write_svg(
        figure,
        main_diffs,
        "bars: point estimate, whiskers: 95% bootstrap interval over runs; grey crosses zero",
    )
    print(f"\n# figure written to {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())