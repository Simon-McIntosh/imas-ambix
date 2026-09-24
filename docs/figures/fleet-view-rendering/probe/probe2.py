import os, pty, select, struct, fcntl, termios, time, sys
ZJ="/home/ITER/mcintos/.local/bin/zellij"
sdir, sess, cols, rows, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
def swing(fd,r,c): fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH",r,c,0,0))
t0=time.monotonic()
pid, fd = pty.fork()
if pid==0:
    os.environ["ZELLIJ_SOCKET_DIR"]=sdir
    os.environ.pop("ZELLIJ_SESSION_NAME",None); os.environ.pop("ZELLIJ",None)
    os.environ["TERM"]="xterm-256color"
    swing(0,rows,cols); os.execv(ZJ,[ZJ,"attach",sess])
swing(fd,rows,cols)
def rd(budget, quiet):
    start=time.monotonic(); first=None; last=start; n=0
    while time.monotonic()-start < budget:
        r,_,_=select.select([fd],[],[],0.02)
        if r:
            try: d=os.read(fd,262144)
            except OSError: break
            if not d: break
            now=time.monotonic()
            if first is None: first=now
            last=now; n+=len(d)
        elif first is not None and time.monotonic()-last > quiet: break
    return first,last,n
f,l,n = rd(60.0, 0.5)
print("attach_first_ms=%.1f attach_settle_ms=%.1f attach_bytes=%d" % (
  (f-t0)*1000 if f else -1, (l-t0)*1000, n))
if mode=="attach_resize":
    rd(2.0,0.5)
    tr=time.monotonic(); swing(fd,rows,cols-20)
    f2,l2,n2 = rd(60.0,0.5)
    print("resize_first_ms=%.1f resize_settle_ms=%.1f resize_bytes=%d" % (
      (f2-tr)*1000 if f2 else -1,(l2-tr)*1000,n2))
try: os.kill(pid,9); os.waitpid(pid,0)
except Exception: pass
