#!/usr/bin/env python3
"""Measure zellij screen-thread stall on attach and on resize, from a client pty.

Instrument: a client's time-to-first-byte after attach, and after a window-size
change.  The server's screen thread must complete the pane relayout and render
before it can emit anything to the new client, so a stalled screen thread
delays that first byte.  Time-to-first-byte therefore measures the stall the
server-side log can only report coarsely (its threshold is one second).

Usage: probe.py <socket-dir> <session> <cols> <rows> <mode>
  mode = attach          attach, measure first byte, detach
  mode = attach_resize   attach, settle, resize by (cols-20), measure first byte
"""
import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time

ZJ = "/home/ITER/mcintos/.local/bin/zellij"


def set_win(fd, rows, cols):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def spawn_attach(session, sockdir, cols, rows):
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["ZELLIJ_SOCKET_DIR"] = sockdir
        os.environ.pop("ZELLIJ_SESSION_NAME", None)
        os.environ.pop("ZELLIJ", None)
        os.environ["TERM"] = "xterm-256color"
        set_win(0, rows, cols)
        os.execv(ZJ, [ZJ, "attach", session])
    set_win(fd, rows, cols)
    return pid, fd


def drain(fd, budget_s, quiet_s=0.15, first_only=False):
    """Return (t_first, t_last, nbytes) relative to call, or None first."""
    start = time.monotonic()
    first = None
    last = start
    nbytes = 0
    while time.monotonic() - start < budget_s:
        r, _, _ = select.select([fd], [], [], 0.02)
        if r:
            try:
                data = os.read(fd, 262144)
            except OSError:
                break
            if not data:
                break
            if first is None:
                first = time.monotonic()
            last = time.monotonic()
            nbytes += len(data)
            if first_only:
                break
        elif first is not None and time.monotonic() - last > quiet_s:
            break
    return (first, last, nbytes)


def kill(pid, fd):
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.kill(pid, 15)
    except OSError:
        pass
    time.sleep(0.2)
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    os.waitpid(pid, os.WNOHANG)


def main():
    sockdir, session = sys.argv[1], sys.argv[2]
    cols, rows = int(sys.argv[3]), int(sys.argv[4])
    mode = sys.argv[5]
    t0 = time.monotonic()
    pid, fd = spawn_attach(session, sockdir, cols, rows)
    first, last, nb = drain(fd, 120.0, first_only=(mode == "attach"))
    attach_ms = (first - t0) * 1000.0 if first is not None else None
    print("attach_ms=%.1f attach_bytes=%d" % (
        attach_ms if attach_ms is not None else -1, nb))
    resize_ms = None
    if mode == "attach_resize":
        drain(fd, 3.0, quiet_s=0.4)          # settle
        ncols = cols - 20
        t_r = time.monotonic()
        set_win(fd, rows, ncols)
        r_first, r_last, r_nb = drain(fd, 120.0, first_only=True)
        resize_ms = (r_first - t_r) * 1000.0 if r_first is not None else None
        print("resize_ms=%.1f resize_bytes=%d new_cols=%d" % (
            resize_ms if resize_ms is not None else -1, r_nb, ncols))
    kill(pid, fd)
    sys.stdout.flush()


if __name__ == "__main__":
    main()