import os, pty, select, struct, fcntl, termios, time, sys
ZJ="/home/ITER/mcintos/.local/bin/zellij"
sdir, sess = sys.argv[1], sys.argv[2]
def swing(fd,r,c): fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH",r,c,0,0))
t0=time.monotonic()
pid, fd = pty.fork()
if pid==0:
    os.environ["ZELLIJ_SOCKET_DIR"]=sdir
    os.environ.pop("ZELLIJ_SESSION_NAME",None); os.environ.pop("ZELLIJ",None)
    os.environ["TERM"]="xterm-256color"
    swing(0,40,120); os.execv(ZJ,[ZJ,"attach",sess])
swing(fd,40,120)
start=time.monotonic(); n=0; buf=b""; replied=set()
while time.monotonic()-start < 25:
    r,_,_=select.select([fd],[],[],0.05)
    if r:
        try: d=os.read(fd,262144)
        except OSError: break
        if not d: break
        buf+=d; n+=len(d)
        # answer common terminal queries so the client can proceed
        for q,a in ((b"\x1b[6n", b"\x1b[1;1R"), (b"\x1b[c", b"\x1b[?62;c"),
                    (b"\x1b[>c", b"\x1b[>0;95;0c"), (b"\x1b[?u", b"\x1b[?0u"),
                    (b"\x1b[5n", b"\x1b[0n")):
            if q in d and q not in replied:
                os.write(fd, a); replied.add(q)
                print("replied to %r at %.2fs" % (q, time.monotonic()-t0))
print("TOTAL %d bytes in 25s; first chunk:" % n)
print(repr(buf[:400]))
os.kill(pid,9); os.waitpid(pid,0)
