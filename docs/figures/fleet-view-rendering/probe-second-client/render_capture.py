#!/usr/bin/env python3
"""Render a captured zellij client stream through a VT emulator (pyte).

The capture is one client's whole byte stream; the marks file records the
offsets at which the interesting moments begin. Each snapshot is written as
plain text and the interesting rows are printed, so a tab-bar row or a
duplicated line can be read off the screen rather than inferred from bytes.

Usage: render_capture.py <cap.bin> <marks.json> <mode> <out_prefix>

The width change during the live-app scenario is applied to the screen with
pyte's own resize, which is what a real terminal does at that moment.
"""
import json
import sys

import pyte


class TolerantScreen(pyte.Screen):
    """A screen that ignores the private-mode replies zellij emits.

    zellij sends `CSI ? ... $ p` (device status) and mode queries; pyte's
    handler rejects the `private` keyword those arrive with, which would abort
    the render. The reply is a query answer, not screen content, so dropping it
    changes nothing that is on screen.
    """

    def report_device_status(self, *a, **kw):
        pass

    def set_mode(self, *a, **kw):
        pass

    def reset_mode(self, *a, **kw):
        pass


def snap(screen):
    return [row.rstrip() for row in screen.display]


FRAME = set("│─┌┐└┘┬┴├┤┼╭╮╰╯━┏┓┗┛┃┣┫┳┻╋")


def is_frame(row):
    return not (set(row.replace(" ", "")) - FRAME)


def dup_text(rows):
    """Adjacent identical rows that carry content, not just frame borders."""
    return [d for d in dup_lines(rows) if not is_frame(d[1]) if d[1]]


def dup_lines(rows):
    """Adjacent identical non-empty rows, the shape a reflow garble takes."""
    out = []
    for i in range(1, len(rows)):
        if rows[i] and rows[i] == rows[i - 1]:
            out.append((i, rows[i]))
    return out


def emit(prefix, name, rows):
    path = "%s.%s.txt" % (prefix, name)
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("snapshot=%s rows=%d dup_text=%d dup_frame_rows=%d path=%s" % (
        name, len(rows), len(dup_text(rows)), len(dup_lines(rows)), path))
    return rows


def main():
    cap, marks_path, mode, prefix = sys.argv[1:5]
    data = open(cap, "rb").read()
    marks = json.load(open(marks_path))
    rows_n = marks.get("rows", 30)

    if mode == "multi":
        cols = marks["width_first"]
        screen = TolerantScreen(cols, rows_n)
        stream = pyte.ByteStream(screen)
        stream.feed(data[:marks["mark_before_second"]])
        before = emit(prefix, "before", snap(screen))
        stream.feed(data[marks["mark_before_second"]:marks["mark_after_second"]])
        after = emit(prefix, "after", snap(screen))
        print("tabbar_before=%r" % before[0])
        print("tabbar_after=%r" % after[0])
        print("tabbar_equal=%s" % (before[0] == after[0]))
        print("dups_before=%r" % (dup_text(before),))
        print("dups_after=%r" % (dup_text(after),))
    elif mode == "live":
        w0, w1 = marks["width_before"], marks["width_after"]
        screen = TolerantScreen(w0, rows_n)
        stream = pyte.ByteStream(screen)
        stream.feed(data[:marks["mark_before_resize"]])
        before = emit(prefix, "before", snap(screen))
        print("tabbar_before=%r" % before[0])
        print("dups_before_n=%d" % len(dup_lines(before)))
        screen.resize(rows_n, w1)
        stream.feed(data[marks["mark_before_resize"]:marks["mark_end"]])
        after = emit(prefix, "after", snap(screen))
        print("tabbar_after=%r" % after[0])
        print("dups_after_n=%d" % len(dup_lines(after)))
        print("dups_after=%r" % (dup_lines(after)[:12],))
    else:
        sys.exit("unknown mode %s" % mode)


if __name__ == "__main__":
    main()