"""Measure the gate wait bound over the fixed seed set, and check the fragment.

``run`` simulates the calibrated lane at the bound's demonstration margin for
each seed, both arms (bound enabled and disabled), and then the think-time-drop
run in both arms, writing one log the fragment cites for every figure it states.

``check`` reads that log and the fragment beside it and fails unless every
figure the fragment states is the log's, and unless every log the fragment
cites is a file in the repository.

Each five-minute window's figure is cut with ``percentile_or_none``, the one
nearest-rank estimator the lane's own figures use, so the log agrees with what
the gate publishes for the same record. ``check`` asserts that provenance: a
second nearest-rank rule agreeing today is indistinguishable from the owner's
by any figure it produces, so the check reads where the callable lives and
fails if a local estimator has grown back beside the import.

usage: bound_seeds.py run <log>      (on a compute partition)
       bound_seeds.py check <log> <fragment>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from imas_ambix.agent import lane_utilisation
from imas_ambix.agent.lane_sim import LaneSim, calibrate_rows, read_receipts
from imas_ambix.agent.lane_utilisation import percentile_or_none
from imas_ambix.agent.router import DEFAULT_WAIT_BOUND_SECONDS

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
RECEIPTS = REPO / "tests" / "data" / "lane_receipts" / "2026-11-18T0445-0640.jsonl.gz"
WINDOW = ("2026-11-18T04:45:00Z", "2026-11-18T06:40:00Z")

MARGIN = 2.6
SEEDS = tuple(range(10))
HOURS = 4.5
WARM_UP_S = 1800.0
SETTLED_S = WARM_UP_S + 600.0
WINDOW_S = 300.0
LIMIT_S = 45.0

THINK_DROP_S = 7200.0
THINK_DROP_SEED = 0


def _window_p90s(sim: LaneSim) -> list[tuple[float, float | None]]:
    windows = []
    lo = WARM_UP_S
    while lo + WINDOW_S <= sim.end:
        waits = [wait for at, wait in sim.wait_events if lo <= at < lo + WINDOW_S]
        windows.append((lo, percentile_or_none(waits, 0.9)))
        lo += WINDOW_S
    return windows


def _estimator_problem() -> str | None:
    """Why the window figure may not be the lane's estimator, if it is not.

    A nearest-rank rule written here would agree with the owner's on every
    input, so no figure the log holds can tell the two apart: only where the
    callable lives can. The check asks that, and reads this file for a
    percentile body grown back beside the import.
    """
    if percentile_or_none.__module__ != lane_utilisation.__name__:
        return (
            "the window figure is not cut with the lane's estimator: "
            f"{percentile_or_none.__module__}.{percentile_or_none.__qualname__}"
        )
    if "percentile_or_none" not in _window_p90s.__code__.co_names:
        return "the window figure does not call percentile_or_none"
    source = Path(__file__).read_text(encoding="utf-8")
    # This function's own text names the token it looks for.
    skip = {"_estimator_problem"}
    for match in re.finditer(r"^def (\w+)\(", source, re.MULTILINE):
        name = match.group(1)
        if name in skip:
            continue
        rest = source[match.end() :]
        end = re.search(r"^(?:def|class) ", rest, re.MULTILINE)
        body = rest[: end.start()] if end else rest
        if "sorted(" in body:
            return (
                f"this script defines its own percentile body beside the lane's: {name}"
            )
    return None


def _settled(sim: LaneSim) -> list[float]:
    return [p for lo, p in _window_p90s(sim) if lo >= SETTLED_S and p is not None]


def _windows_over(settled: list[float]) -> int:
    """Settled windows strictly above the limit the pass condition names."""
    return sum(1 for p in settled if p > LIMIT_S)


def _closed(sim: LaneSim) -> int:
    return sum(1 for d in sim.documents if d["worker_slots_reason"] == "gate_wait")


def _run(calibration, *, bound, seed, margin=MARGIN, think_scale=None):
    sim = LaneSim(
        calibration,
        seed=seed,
        hours=HOURS,
        margin=margin,
        wait_bound_seconds=bound,
        think_scale=think_scale,
    )
    sim.run()
    return sim


def run(log_path: Path) -> int:
    calibration = calibrate_rows(read_receipts(RECEIPTS), WINDOW)
    lines = [
        "# gate wait bound, ten fixed seeds, both arms",
        f"# module: {LaneSim.__module__}",
        f"# module_file: {sys.modules[LaneSim.__module__].__file__}",
        f"# cwd: {Path.cwd()}",
        f"# calibration: {RECEIPTS.relative_to(REPO)} window {WINDOW[0]}..{WINDOW[1]}",
        f"# margin: {MARGIN} hours: {HOURS} warm_up_s: {WARM_UP_S} "
        f"settled_from_s: {SETTLED_S} window_s: {WINDOW_S}",
        f"# bound_seconds: {DEFAULT_WAIT_BOUND_SECONDS}",
        "# columns: seed bounded_p90_max unbounded_p90_max "
        "unbounded_windows_over_45 bounded_windows_over_45 "
        "bounded_closed_documents",
    ]
    for seed in SEEDS:
        bounded = _run(calibration, bound=DEFAULT_WAIT_BOUND_SECONDS, seed=seed)
        unbounded = _run(calibration, bound=None, seed=seed)
        bounded_settled = _settled(bounded)
        unbounded_settled = _settled(unbounded)
        if not bounded_settled or not unbounded_settled:
            print(f"seed {seed}: no settled window", file=sys.stderr)
            return 1
        lines.append(
            f"seed {seed} "
            f"{max(bounded_settled):.2f} "
            f"{max(unbounded_settled):.2f} "
            f"{_windows_over(unbounded_settled)} "
            f"{_windows_over(bounded_settled)} "
            f"{_closed(bounded)}"
        )

    # The think-time-drop run: think time divided by three at the two-hour mark,
    # at the default margin, in both arms.
    def drop(t: float) -> float:
        return 1.0 / 3.0 if t >= THINK_DROP_S else 1.0

    lines.append(
        f"# think_drop: margin 1.5 think/3 at {THINK_DROP_S:g} s seed {THINK_DROP_SEED}"
    )
    for name, bound in (
        ("bound_enabled", DEFAULT_WAIT_BOUND_SECONDS),
        ("bound_disabled", None),
    ):
        sim = _run(
            calibration,
            bound=bound,
            seed=THINK_DROP_SEED,
            margin=1.5,
            think_scale=drop,
        )
        settled = _settled(sim)
        if not settled:
            print(f"think drop {name}: no settled window", file=sys.stderr)
            return 1
        lines.append(
            f"think_drop {name} {max(settled):.2f} "
            f"{_windows_over(settled)} {_closed(sim)}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def _log_figures(text: str) -> dict:
    figures = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "seed":
            figures[line.split()[1]] = {
                "bounded": parts[2],
                "unbounded": parts[3],
                "unbounded_over": parts[4],
                "bounded_over": parts[5],
            }
        elif parts[:2] == ["think_drop", "bound_enabled"]:
            figures["think_drop"] = {"enabled": parts[2]}
        elif parts[:2] == ["think_drop", "bound_disabled"]:
            figures.setdefault("think_drop", {})["disabled"] = parts[2]
    return figures


def check(fragment: Path, log_path: Path) -> int:
    problems = []
    estimator = _estimator_problem()
    if estimator:
        problems.append(estimator)
    text = fragment.read_text(encoding="utf-8")
    for cited in re.findall(r"<code>(logs/[^<]+)</code>", text):
        if not (fragment.parent / cited).exists():
            problems.append(f"cited log absent from the repository: {cited}")
    if not log_path.exists():
        problems.append(f"the log the figures are read from is absent: {log_path}")
        print("\n".join(problems))
        return 1
    figures = _log_figures(log_path.read_text(encoding="utf-8"))
    rows = re.findall(
        r"<tr><td>(\d+)</td><td>([\d.]+) s</td><td>([\d.]+) s</td>"
        r"<td>(yes|no)</td></tr>",
        text,
    )
    if len(rows) != len(SEEDS):
        problems.append(
            f"the fragment states {len(rows)} seed rows, the log {len(SEEDS)}"
        )
    for seed, bounded, unbounded, crosses in rows:
        logged = figures.get(seed)
        if logged is None:
            problems.append(f"seed {seed}: not in the log")
            continue
        if float(logged["bounded"]) != float(bounded):
            problems.append(
                f"seed {seed} bounded: fragment {bounded} s, log {logged['bounded']} s"
            )
        if float(logged["unbounded"]) != float(unbounded):
            problems.append(
                f"seed {seed} unbounded: fragment {unbounded} s, "
                f"log {logged['unbounded']} s"
            )
        logged_crosses = "yes" if int(logged["unbounded_over"]) > 0 else "no"
        if logged_crosses != crosses:
            problems.append(
                f"seed {seed} crosses 45 s: fragment {crosses}, log {logged_crosses}"
            )
        if int(logged["bounded_over"]) != 0:
            problems.append(
                f"seed {seed}: the bounded arm crosses 45 s in "
                f"{logged['bounded_over']} settled window(s)"
            )
    drops = re.findall(r"worst settled window,\s+([\d.]+) s", text)
    if len(drops) != 1:
        problems.append(
            "the fragment does not state the think-time-drop run's worst window"
        )
    elif "think_drop" in figures:
        for arm in ("enabled", "disabled"):
            value = figures["think_drop"].get(arm)
            if value is not None and float(value) != float(drops[0]):
                problems.append(
                    f"think drop {arm}: fragment {drops[0]} s, log {value} s"
                )
    else:
        problems.append("the log holds no think-time-drop run")
    print("\n".join(problems) if problems else "the fragment's figures are the log's")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "check"))
    ap.add_argument("log")
    ap.add_argument("fragment", nargs="?")
    args = ap.parse_args()
    if args.mode == "run":
        return run(Path(args.log))
    if not args.fragment:
        ap.error("check needs the fragment path")
    return check(Path(args.fragment), Path(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
