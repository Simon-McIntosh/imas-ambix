import os
import pty
import select
import struct
import fcntl
import termios
import time
import sys

ZJ = "/home/ITER/mcintos/.local/bin/zellij"
sdir, sess = sys.argv[1], sys.argv[2]
cols, rows = int(sys.argv[3]), int(sys.argv[4])
mode = sys.argv[5]
out = sys.argv[6] if len(sys.argv) > 6 else None

QUERIES = [
    (b"\x1b[6n", b"\x1b[1;1R"),
    (b"\x1b[c", b"\x1b[?62;1;2;6;9;15;18;21;22c"),
    (b"\x1b[>c", b"\x1b[>0;95;0c"),
    (b"\x1b[?u", b"\x1b[?0u"),
    (b"\x1b[5n", b"\x1b[0n"),
    (b"\x1b]11;?", b"\x1b]11;rgb:0000/0000/0000\x1b\\"),
    (b"\x1b]10;?", b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\"),
]


def swing(fd, r, c):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", r, c, 0, 0))


def mkclient(c, r):
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["ZELLIJ_SOCKET_DIR"] = sdir
        os.environ.pop("ZELLIJ_SESSION_NAME", None)
        os.environ.pop("ZELLIJ", None)
        os.environ["TERM"] = "xterm-256color"
        swing(0, r, c)
        os.execv(ZJ, [ZJ, "attach", sess])
    swing(fd, r, c)
    return pid, fd


def read_until(fd, budget, quiet):
    """Read while answering terminal queries; return (t_first, t_last, n, buf)."""
    start = time.monotonic() if False else time.monotonic()
    first = None
    last = time.monotonic()
    n = 0
    buf = b""
    answered = set()
    while time.monotonic() - start < budget:
        rd, _, _ = select.select([fd], [], [], 0.02)
        if rd:
            try:
                d = os.read(fd, 262144)
            except OSError:
                break
            if not d:
                break
            now = time.monotonic()
            if first is None:
                first = now
            last = now
            n += len(d)
            buf += d
            for q, a in QUERIES:
                if q in d and q not in answered:
                    os.write(fd, a)
                    answered.add(q)
        elif first is not None and time.monotonic() - last > quiet:
            break
    return first, last, n, buf


def stop(pid, fd):
    try:
        os.close(fd)
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


t0 = time.monotonic()
pid, fd = mkclient(cols, rows)
f, l, n, buf = read_until(fd, 60.0, 0.5)
print("attach_first_ms=%d attach_render_ms=%d attach_bytes=%d" % (
    (f - t0) * 1000 if f else -1, (l - t0) * 1000, n))

if mode == "timed":
    read_until(fd, 2.0, 0.5) if False else read_until(fd, 2.0, 0.5)
    tr = time.monotonic()
    swing(fd, rows, cols - 20)
    f2, l2, n2, _ = read_until(fd, 60.0, 0.5)
    print("resize"[:0] + "resize_first_ms=%d resize_render_ms=%d resize_bytes=%d new_cols=%d" % (
        (f2 - tr) * 1000 if f2 else -1, (l2 - tr) * 1000, n2, cols - 20))

if mode == "twoclient":
    read_until(fd, 2.0, 0.5)
    base = len(buf)
    t2 = time.monotonic()
    p2, fd2 = mkclient(cols - 20, rows)
    f3, l3, n3, buf3 = read_until(fd, 10.0, 0.5)
    buf += buf3
    print("second_attach_first_ms=%d second_attach_bytes=%d" % (
        (f3 - t2) * 1000 if f3 else -1, n3))
    stop(p2, fd2)
    if out:
        open(out, "wb").write(buf[base:])
        print("first_client_capture=%s" % out)

stop(pid, fd)