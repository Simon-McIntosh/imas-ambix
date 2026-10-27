"""The on-node sampler, run as one long-lived SLURM step.

``python -m imas_ambix.agent.gpu_watch.sampler`` streams
:meth:`imas_ambix.agent.node_probe.NodeProbe.stream` as one JSON object per
line. The launcher starts it with ``srun --jobid <job> --overlap`` so a watcher
running anywhere on the cluster reads a running server's cards, host and job
table through the allocation that server already holds.

One place owns the readings: this module only frames what ``NodeProbe``
answers, so a card's physical index is resolved once, by the same reader the
recorder uses, rather than by a copy of the probe travelling to the node. A
line is flushed as it is written, so the cadence the launcher observes is the
cadence the probe took.
"""

from __future__ import annotations

import argparse
import json
import sys

from imas_ambix.agent import node_probe


def main(argv: list[str] | None = None) -> int:
    """Emit one sample per interval until *count* samples or forever."""
    parser = argparse.ArgumentParser(
        prog="python -m imas_ambix.agent.gpu_watch.sampler",
        description="Stream the node probe's samples as JSON lines.",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="seconds between samples"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="samples to emit before exiting; 0 streams until interrupted",
    )
    args = parser.parse_args(argv)

    probe = node_probe.NodeProbe()
    count = args.count if args.count > 0 else None
    try:
        for sample in probe.stream(args.interval, count=count):
            sys.stdout.write(json.dumps(sample, default=str) + "\n")
            sys.stdout.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        # A reader that closed the pipe is the watcher going away; the step is
        # being reaped, so ending quietly is the correct answer.
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
