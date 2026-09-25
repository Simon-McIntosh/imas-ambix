#!/usr/bin/env python3
"""Plot the second-client probe timings and the stall-line deltas.

Left panel: attach, second attach and resize milliseconds per scenario, read
from the marks files the harness writes. Right panel: the stall-line count
before and after each run, which is 33 -> 33 in every case, with the run that
ended in a server panic marked, because that failure is not a stall and so
does not appear in the counted line at all.

Usage: plot_timings.py <evidence-dir> <out.png>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("zj-second-4tab-attached", "4 tabs, tabs\nopened attached"),
    ("zj-second-1tab-attached", "1 tab\n(control)"),
    ("zj-second-live", "live claude\npane, resize"),
]
STALL = [("4tab-attached", 33, 33, False), ("4tab-pre", 33, 33, True),
         ("1tab control", 33, 33, False), ("live resize", 33, 33, False)]


def load(ev, stem):
    return json.load(open(os.path.join(ev, stem + ".marks.json")))


def main():
    ev, out = sys.argv[1], sys.argv[2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    labels, groups = [], {"attach": [], "second attach": [], "resize": []}
    for stem, label in RUNS:
        m = load(ev, stem)
        labels.append(label)
        groups["attach"].append(m.get("attach_ms", 0))
        groups["second attach"].append(m.get("second_attach_ms", 0) or 0)
        groups["resize"].append(max(m.get("resize_ms", 0), 0))

    idx = range(len(labels))
    width = 0.26
    for i, (name, vals) in enumerate(groups.items()):
        ax1.bar([x + i * width for x in idx], vals, width, label=name)
        for x, v in zip(idx, vals):
            if v:
                ax1.text(x + i * width, v + 60, str(v), ha="center", fontsize=8)
    ax1.set_xticks([x + width for x in idx])
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("milliseconds")
    ax1.set_title("Client render time by scenario")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 4200)

    names = [s[0] for s in STALL]
    before = [s[1] for s in STALL]
    after = [s[2] for s in STALL]
    y = range(len(names))
    ax2.barh([v + 0.2 for v in y], before, 0.38, label="before run")
    ax2.barh([v - 0.2 for v in y], after, 0.38, label="after run")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlim(0, 40)
    ax2.set_xlabel("stall lines in the server log")
    ax2.set_title("Stall lines before and after each run (all deltas 0)")
    for i, s in enumerate(STALL):
        if s[3]:
            ax2.text(34.5, i, "server panicked", fontsize=8, va="center",
                     color="crimson")
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()