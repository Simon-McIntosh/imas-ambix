"""Re-derive the clive lane's usable input window from the live endpoint.

reckon's flight config takes ``usable_input_window`` as a literal, and it gates
a pre-dispatch context-fit refusal: absence disables that fence rather than
making it unbounded in a useful way, so the value cannot simply be dropped. It
is therefore a copy of a deployment property, and copies rot -- this one was
stale within the hour when it was first set, and stale again when the serve
took a 204,800-token context cap.

This makes refreshing it one command rather than remembered arithmetic:
read what the endpoint advertises, subtract the launcher's output reservation,
and write the result back.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_ORIGIN = "http://98dci4-gpu-0003:18802"
DEFAULT_CONFIG = Path.home() / ".config" / "reckon" / "flight.yaml"
LAUNCHER = Path("/work/projects/imas_gpu/agents/clive")


def advertised_window(origin: str) -> int:
    """Return max_model_len as the endpoint itself reports it."""
    with urllib.request.urlopen(
        f"{origin.rstrip('/')}/v1/models", timeout=30
    ) as response:
        payload = json.loads(response.read())
    cards = payload.get("data") or []
    if not cards:
        raise SystemExit(f"{origin} advertises no models")
    window = cards[0].get("max_model_len")
    if not isinstance(window, int) or window < 1:
        raise SystemExit(f"{origin} reports an unusable max_model_len: {window!r}")
    return window


def output_reservation(launcher: Path) -> int:
    """Read the launcher's own output reservation rather than assuming it."""
    match = re.search(r"OUTPUT_RESERVATION=(\d+)", launcher.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"no OUTPUT_RESERVATION found in {launcher}")
    return int(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument(
        "--write", action="store_true", help="Apply the change; default is a dry run."
    )
    args = parser.parse_args(argv)

    window = advertised_window(args.origin)
    reservation = output_reservation(args.launcher)
    usable = window - reservation
    if usable < 1:
        raise SystemExit(
            f"output reservation {reservation} exceeds the advertised window {window}"
        )

    text = args.config.read_text(encoding="utf-8")
    current = re.search(r"^(\s*)usable_input_window:\s*(\d+)\s*$", text, re.MULTILINE)
    if current is None:
        raise SystemExit(f"no usable_input_window key found in {args.config}")
    declared = int(current.group(2))

    print(f"  endpoint advertises : {window}")
    print(f"  output reservation  : {reservation}")
    print(f"  usable input window : {usable}")
    print(f"  config declares     : {declared}")
    if declared == usable:
        print("  in step — nothing to do")
        return 0
    verdict = (
        "OVERSTATED — nodes will be sized past what the serve accepts"
        if declared > usable
        else "understated — nodes are sized conservatively"
    )
    print(f"  DRIFT: {verdict}")
    if not args.write:
        print("  dry run; pass --write to apply")
        return 1
    args.config.write_text(
        text[: current.start(2)] + str(usable) + text[current.end(2) :],
        encoding="utf-8",
    )
    print(f"  written: {declared} -> {usable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
