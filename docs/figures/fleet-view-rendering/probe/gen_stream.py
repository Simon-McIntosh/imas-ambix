#!/usr/bin/env python3
"""Build a Claude Code byte-stream pane payload of a required line count.

The transcript body is taken from a real recorded Claude Code terminal stream
(`script -q -c 'claude -p ...'`), repeated to reach the target line count.
Real Claude Code output is used because its line length and SGR usage are what
a width change must reflow; the recorded body is cycled rather than invented.

Usage: gen_stream.py <target_lines> <source_raw> [plain]
"""
import re
import sys

BODY_RE = re.compile(r"^\s*\d+\.\s+\S")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def body_lines(raw):
    lines = []
    for line in raw.split("\n"):
        line = line.rstrip("\r")
        plain = ANSI_RE.sub("", line)
        if BODY_RE.match(plain) and len(plain) > 40:
            lines.append(line)
    return lines


def main():
    target = int(sys.argv[1])
    src = sys.argv[2]
    colour = (len(sys.argv) < 4) or (sys.argv[3] != "plain")
    with open(src, "r", errors="replace") as fh:
        raw = fh.read()
    body = body_lines(raw)
    if not body:
        sys.exit("no Claude Code body lines found in %s" % src)
    out = ["\x1b[2J\x1b[H"]
    i = 0
    while i < target:
        line = body[i % len(body)]
        if colour and i % 3 == 0:
            out.append("\x1b[38;5;%dm%s\x1b[0m\r\n" % (16 + (i * 7) % 200, line))
        else:
            out.append("%s\r\n" % line)
        i += 1
    out.append("\x1b[H")
    sys.stdout.write("".join(out))
    sys.stderr.write("emitted_lines=%d body_lines=%d\n" % (target, len(body)))


if __name__ == "__main__":
    main()