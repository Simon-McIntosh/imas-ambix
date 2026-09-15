"""Reconcile clive's deployment-derived flight settings from the live endpoint.

reckon's flight config takes ``usable_input_window`` as a literal, and it gates
a pre-dispatch context-fit refusal: absence disables that fence rather than
making it unbounded in a useful way, so the value cannot simply be dropped. It
is therefore a copy of a deployment property, and copies rot -- this one was
stale within the hour when it was first set, and stale again when the serve
took a 204,800-token context cap.

This makes refreshing the full deployment identity one command: read what the
endpoint advertises, subtract the launcher's output reservation, and write the
result back. The served model name is copied to both ``model`` and ``alias``:
Reckon's ledger displays the model when no alias is supplied, so the endpoint
name is the only non-guessed alias for a rotating deployment.
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


def advertised_deployment(origin: str) -> tuple[str, int]:
    """Return the served model name and max_model_len from the endpoint."""
    with urllib.request.urlopen(
        f"{origin.rstrip('/')}/v1/models", timeout=30
    ) as response:
        payload = json.loads(response.read())
    cards = payload.get("data") or []
    if not cards:
        raise SystemExit(f"{origin} advertises no models")
    card = cards[0]
    if not isinstance(card, dict):
        raise SystemExit(f"{origin} advertises an invalid model card")
    model = card.get("id")
    if (
        not isinstance(model, str)
        or not model.strip()
        or any(ord(character) < 32 for character in model)
    ):
        raise SystemExit(f"{origin} reports an unusable model id: {model!r}")
    window = card.get("max_model_len")
    if not isinstance(window, int) or window < 1:
        raise SystemExit(f"{origin} reports an unusable max_model_len: {window!r}")
    return model, window


def output_reservation(launcher: Path) -> int:
    """Read the launcher's own output reservation rather than assuming it."""
    match = re.search(r"OUTPUT_RESERVATION=(\d+)", launcher.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"no OUTPUT_RESERVATION found in {launcher}")
    return int(match.group(1))


def clive_block(text: str) -> tuple[int, int]:
    """Return the source range of the ``backends.clive`` mapping."""
    backends = re.search(r"^(?P<indent>\s*)backends:\s*(?:#.*)?$", text, re.MULTILINE)
    if backends is None:
        raise SystemExit("no backends mapping found in flight config")
    parent_indent = len(backends.group("indent"))
    backend_end = re.compile(rf"^[ ]{{{parent_indent}}}\S.*$", re.MULTILINE).search(
        text, backends.end()
    )
    backend_text_end = backend_end.start() if backend_end else len(text)
    clive = re.compile(r"^(?P<indent>[ ]+)clive:\s*(?:#.*)?$", re.MULTILINE).search(
        text, backends.end(), backend_text_end
    )
    if clive is None:
        raise SystemExit("no clive backend found in flight config")
    sibling_indent = len(clive.group("indent"))
    next_sibling = re.compile(rf"^[ ]{{{sibling_indent}}}\S.*$", re.MULTILINE).search(
        text, clive.end(), backend_text_end
    )
    return clive.end(), next_sibling.start() if next_sibling else backend_text_end


def replace_clive_value(
    text: str, block_start: int, block_end: int, key: str, value: str
) -> tuple[str, str]:
    """Replace one scalar in the clive block and return its prior spelling."""
    block = text[block_start:block_end]
    current = re.search(
        rf"^(?P<indent>\s*){re.escape(key)}:\s*(?P<value>[^#\n]*?)"
        r"(?P<suffix>\s*(?:#.*)?)$",
        block,
        re.MULTILINE,
    )
    if current is None:
        raise SystemExit(f"no clive {key} key found in flight config")
    declared = current.group("value").strip()
    replacement = current.group("indent") + key + ": " + value + current.group("suffix")
    updated_block = block[: current.start()] + replacement + block[current.end() :]
    return text[:block_start] + updated_block + text[block_end:], declared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument(
        "--write", action="store_true", help="Apply the change; default is a dry run."
    )
    args = parser.parse_args(argv)

    model, window = advertised_deployment(args.origin)
    reservation = output_reservation(args.launcher)
    usable = window - reservation
    if usable < 1:
        raise SystemExit(
            f"output reservation {reservation} exceeds the advertised window {window}"
        )

    text = args.config.read_text(encoding="utf-8")
    replacements = {
        "model": json.dumps(model),
        "alias": json.dumps(model),
        "usable_input_window": str(usable),
    }
    declared: dict[str, str] = {}
    for key, value in replacements.items():
        block_start, block_end = clive_block(text)
        text, declared[key] = replace_clive_value(
            text, block_start, block_end, key, value
        )

    print(f"  endpoint advertises : {model} ({window})")
    print(f"  output reservation  : {reservation}")
    print(f"  usable input window : {usable}")
    changes = [
        (key, before, after)
        for key, after in replacements.items()
        if (before := declared[key]) != after
    ]
    if not changes:
        print("  in step — nothing to do")
        return 0
    for key, before, after in changes:
        print(f"  {key:20}: {before} -> {after}")
    if not args.write:
        print("  dry run; pass --write to apply")
        return 1
    args.config.write_text(text, encoding="utf-8")
    print(f"  written: {len(changes)} clive deployment values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
