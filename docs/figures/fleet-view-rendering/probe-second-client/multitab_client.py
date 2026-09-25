#!/usr/bin/env python3
"""Drive zellij clients in ptys and capture the first client's byte stream.

Three scenarios, each run against a throwaway session and each writing the
first client's whole stream plus the byte offsets of the interesting moments
to a capture file and a marks file:

  multi  two clients at different widths attach to the same session; the
         first client's stream is captured across the second attach and a
         subsequent resize
  live   one client attaches to a session holding a live application pane,
         then the client is resized between two widths

Client render time is measured from the client's own quiet period: the moment
the client's output falls quiet for `quiet` seconds after it begins. The child
answers the terminal queries a real terminal answers automatically; without
that a zellij client renders nothing (see the note's harness-defect section).
"""
import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time

ZJ = "/home/ITER/mcintos/.local/bin/zellij"

QUERIES = [
    (b"\x1b[6n", b"\x1b[1;1R"),
    (b"\x1b[c", b"\x1b[?62;1;2;6;9;15;18;21;22c"),
    (b"\x1b[>c", b"\x1b[>0;95;0c"),
    (b"\x1b[?u", b"\x1b[?0u"),
    (b"\x1b[5n", b"\x1b[0n"),
    (b"\x1b]11;?", b"\x1b]11;rgb:0000/0000/0000\x1b\\"),
    (b"\x1b]10;?", b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\"),
]


def swing(fd, rows, cols):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def mkclient(sdir, sess, cols, rows):
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["ZELLIJ_SOCKET_DIR"] = sdir
        os.environ.pop("ZELLIJ_SESSION_NAME", None)
        os.environ.pop("ZELLIJ", None)
        os.environ["TERM"] = "xterm-256color"
        swing(0, rows, cols)
        os.execv(ZJ, [ZJ, "attach", sess])
    swing(fd, rows, cols)
    return pid, fd


def pump(fds, budget, quiet):
    """Read from every fd; answer queries; per-fd first/last/n/buf."""
    start = time.monotonic()
    st = {fd: {"first": None, "last": None, "n": 0, "buf": bytearray()} for fd in fds}
    answered = {fd: set() for fd in fds}
    last_any = None
    while time.monotonic() - start < budget:
        rd, _, _ = select.select(fds, [], [], 0.02)
        if not rd:
            if last_any is not None and time.monotonic() - last_any > quiet:
                break
            continue
        for fd in rd:
            try:
                d = os.read(fd, 262144)
            except OSError:
                d = b""
            if not d:
                continue
            now = time.monotonic()
            s = st[fd]
            if s["first"] is None:
                s["first"] = now
            s["last"] = now
            s["n"] += len(d)
            s["buf"] += d
            last_any = now
            for q, a in QUERIES:
                if q in d and q not in answered[fd]:
                    os.write(fd, a)
                    answered[fd].add(q)
    return st


def stop(pid, fd):
    for fn in (lambda: os.close(fd), lambda: os.write(fd, b"\x04")):
        try:
            fn()
        except OSError:
            pass
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def ms(t, t0):
    return int((t - t0) * 1000) if t else -1


def main():
    sdir, sess, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = sys.argv[4]
    marks_path = sys.argv[5]
    marks = {}
    stream = bytearray()

    if mode == "multi":
        tabs = int(sys.argv[6]) if len(sys.argv) > 6 else 4
        ordering = sys.argv[7] if len(sys.argv) > 7 else "attached"
        cols, rows, second_cols = 100, 30, 60

        def new_tabs(n):
            env = dict(os.environ)
            env["ZELLIJ_SESSION_NAME"] = sess
            env["ZELLIJ_SOCKET_DIR"] = sdir
            for i in range(n):
                subprocess.run([ZJ, "action", "new-tab", "--name", "t%d" % (i + 2)],
                               env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                time.sleep(0.6)

        if ordering == "pre":
            new_tabs(tabs - 1)

        t0 = time.monotonic()
        pidA, fdA = mkclient(sdir, sess, cols, rows)
        st = pump([fdA], 45.0, 1.0)
        stream += st[fdA]["buf"]
        marks["attach_first_ms"] = ms(st[fdA]["first"], t0)
        marks["attach_ms"] = ms(st[fdA]["last"], t0)
        marks["attach_bytes"] = st[fdA]["n"]
        marks["mark_before_tabs"] = len(stream)

        if ordering == "attached":
            new_tabs(tabs - 1)
            stt = pump([fdA], 8.0, 1.0)
            stream += stt[fdA]["buf"]
            marks["tabs_added_bytes"] = stt[fdA]["n"]
        marks["mark_before_second"] = len(stream)
        time.sleep(1.0)

        t1 = time.monotonic()
        pidB, fdB = mkclient(sdir, sess, second_cols, rows)
        st2 = pump([fdA, fdB], 20.0, 0.8)
        first_extra = st2[fdA]["buf"]
        stream += first_extra
        marks["second_attach_first_ms"] = ms(st2[fdB]["first"], t1)
        marks["second_attach_ms"] = ms(st2[fdB]["last"], t1)
        marks["second_attach_bytes"] = st2[fdB]["n"]
        marks["first_extra_bytes"] = len(first_extra)
        marks["first_extra_clear_screen"] = first_extra.count(b"\x1b[2J")
        marks["first_extra_cursor_home"] = first_extra.count(b"\x1b[H")
        marks["mark_after_second"] = len(stream)
        stop(pidB, fdB)
        time.sleep(1.0)

        tr = time.monotonic()
        swing(fdA, rows, cols - 20)
        st3 = pump([fdA], 45.0, 1.0)
        stream += st3[fdA]["buf"]
        marks["resize_first_ms"] = ms(st3[fdA]["first"], tr)
        marks["resize_ms"] = ms(st3[fdA]["last"], tr)
        marks["resize_bytes"] = st3[fdA]["n"]
        marks["mark_end"] = len(stream)
        marks["width_first"] = cols
        marks["width_resized"] = cols - 20
        marks["second_width"] = second_cols
        marks["rows"] = rows
        marks["tabs"] = tabs
        marks["ordering"] = ordering
        stop(pidA, fdA)
        marks["server_error"] = b"Error occurred in server" in bytes(stream)

    elif mode == "live":
        cols, rows, wide = 100, 30, 70
        t0 = time.monotonic()
        pid, fd = mkclient(sdir, sess, cols, rows)
        st = pump([fd], 45.0, 1.2)
        stream += st[fd]["buf"]
        marks["attach_ms"] = ms(st[fd]["last"], t0)
        marks["attach_first_ms"] = ms(st[fd]["first"], t0)
        marks["attach_bytes"] = st[fd]["n"]
        # A fresh workspace shows startup dialogs before the app's own screen;
        # answer them so the resize is measured over the live TUI.
        DIALOGS = [
            (b"trust this folder", b"\x1b[B\r", "trust"),
            (b"use this API key", b"\x1b[A\r", "apikey"),
        ]
        answered = {}
        for _ in range(4):
            s = bytes(stream)
            hit = None
            for needle, keys, name in DIALOGS:
                if name not in answered and needle in s:
                    hit = (keys, name)
                    break
            if hit is None:
                break
            os.write(fd, hit[0])
            stq = pump([fd], 20.0, 1.5)
            stream += stq[fd]["buf"]
            answered[hit[1]] = True
            marks["answered_%s" % hit[1]] = stq[fd]["n"]
        marks["dialogs_answered"] = sorted(answered)
        marks["mark_before_resize"] = len(stream)
        time.sleep(1.0)
        tr = time.monotonic()
        swing(fd, rows, wide)
        st2 = pump([fd], 45.0, 1.2)
        stream += st2[fd]["buf"]
        marks["resize_ms"] = ms(st2[fd]["last"], tr)
        marks["resize_first_ms"] = ms(st2[fd]["first"], tr)
        marks["resize_bytes"] = st2[fd]["n"]
        marks["mark_end"] = len(stream)
        marks["width_before"] = cols
        marks["width_after"] = wide
        marks["rows"] = rows
        stop(pid, fd)

    else:
        sys.exit("unknown mode %s" % mode)

    with open(cap, "wb") as fh:
        fh.write(bytes(stream))
    with open(marks_path, "w") as fh:
        json.dump(marks, fh, indent=2, sort_keys=True)
    print(json.dumps(marks, sort_keys=True))


if __name__ == "__main__":
    main()