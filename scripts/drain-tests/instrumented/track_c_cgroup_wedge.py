"""
CPU-only cgroup-wedge rehearsal: FUSE-freeze D-state wedge without a GPU.
==================================================================================
CPU-only cgroup-wedge rehearsal is the WARM-UP for the coordinated drain window.  It uses NO GPU so it
cannot damage GPU state.  Purpose:

  1. Validate the state-capture harness (observe_state.sh) end-to-end.
  2. Measure the REAL UnkillableStepTimeout / "giving up after N sec" timing
     (never been directly observed; --time 4 min gives a clean trigger).
  3. Rehearse the admin resume procedure BEFORE spending GPU runs on GPU drain reproduction.

Mechanism — fuse-freeze wedge (mechanism: FUSE freeze):
  a. Build a small SquashFS archive containing a multi-MB dummy file.
  b. Mount it with `squashfuse -f` (foreground; this process IS the FUSE daemon).
  c. Start a subprocess that reads the dummy file (continuous dd) — this forces
     sustained FUSE data-block round-trips to the daemon.
  d. While the read is in flight (daemon busy responding), SIGSTOP the daemon
     (os.kill(os.getpid(), signal.SIGSTOP) ... wait, daemon is a child; see below).
  e. With the daemon frozen mid-request the reader's kernel-side wait becomes
     UNINTERRUPTIBLE (D-state, wchan = fuse_simple_request or fuse_dev_do_read).
  f. Main process sleeps forever. SLURM --time fires. SIGTERM → SIGKILL cannot
     reap the D-state reader → slurmstepd "giving up" → node DRAIN.

⚠️  This WILL drain the node. Run ONLY inside the coordinated admin window.
⚠️  No STOP-FILE. No SIGUSR1 trap. Deliberately hardened against clean exit.

Tool detection order (all at runtime — compute node may differ from login node):
  squashfuse + mksquashfs  →  primary (no network, no SSH key, self-contained)
  sshfs                    →  fallback (requires SSH key to localhost, rarely set up)
  bindfs                   →  fallback (bind-mount wrapper, usually absent on HPC)
  none of the above        →  exit 1 cleanly (node NOT drained; no admin action needed)

Recovery (admin, after drain — NO gpu-reset needed, no GPU involved):
  ssh 98dci4-gpu-0003 'ps aux | grep cgroup_wedge'   # confirm gone
  scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
  scontrol show node 98dci4-gpu-0003 | grep State
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Tool detection helpers
# ---------------------------------------------------------------------------


def _cmd_available(name: str) -> bool:
    return shutil.which(name) is not None


def _run_silent(*cmd: str) -> int:
    r = subprocess.run(list(cmd), capture_output=True)
    return r.returncode


def detect_fuse_tool() -> str:
    """
    Return one of 'squashfuse', 'sshfs', 'bindfs', or raise RuntimeError.
    squashfuse is preferred: fully self-contained, no network, no SSH keys.
    """
    if _cmd_available("squashfuse") and _cmd_available("mksquashfs"):
        return "squashfuse"
    if _cmd_available("sshfs"):
        return "sshfs"
    if _cmd_available("bindfs"):
        return "bindfs"
    raise RuntimeError(
        "FATAL: no FUSE tool available (squashfuse+mksquashfs, sshfs, bindfs).\n"
        "  Install squashfs-tools + squashfuse on the compute node, or verify\n"
        "  fuse3 is loaded (modprobe fuse).\n"
        "  Node was NOT drained — no admin action needed."
    )


# ---------------------------------------------------------------------------
# Mount helpers
# ---------------------------------------------------------------------------


def _wait_for_mount(mount_point: str, timeout: float = 10.0) -> None:
    """Poll /proc/mounts until mount_point appears."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with open("/proc/mounts") as f:
            if mount_point in f.read():
                return
        time.sleep(0.1)
    raise RuntimeError(
        f"Mount point {mount_point!r} did not appear in /proc/mounts after {timeout}s"
    )


def mount_squashfuse(work_dir: str, mount_point: str) -> subprocess.Popen:
    """
    Create a SquashFS archive with a 64 MB dummy file, then mount it in
    foreground mode (-f) so the Popen object IS the FUSE daemon.
    Returns the Popen handle (daemon process).
    """
    archive = os.path.join(work_dir, "cgroup_wedge.sqfs")
    dummy_src = os.path.join(work_dir, "dummy")
    os.makedirs(dummy_src, exist_ok=True)
    dummy_file = os.path.join(dummy_src, "payload")

    # Create a 64 MB dummy file to force real data-block round-trips
    print("[cgroup_wedge] creating 64 MB dummy payload ...", flush=True)
    subprocess.run(
        ["dd", "if=/dev/urandom", f"of={dummy_file}", "bs=1M", "count=64"],
        check=True,
        capture_output=True,
    )
    print(f"[cgroup_wedge] dummy payload: {dummy_file}", flush=True)

    print("[cgroup_wedge] building SquashFS archive ...", flush=True)
    subprocess.run(
        ["mksquashfs", dummy_src, archive, "-noappend", "-quiet"],
        check=True,
        capture_output=True,
    )
    print(f"[cgroup_wedge] archive: {archive}", flush=True)

    print(
        f"[cgroup_wedge] mounting with squashfuse -f {archive} {mount_point} ...",
        flush=True,
    )
    # -f: foreground — this process IS the daemon; stdout/stderr stay connected
    daemon = subprocess.Popen(
        ["squashfuse", "-f", archive, mount_point],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give FUSE a moment to register with the kernel
    time.sleep(0.5)
    _wait_for_mount(mount_point, timeout=15.0)
    return daemon


def mount_sshfs(mount_point: str) -> subprocess.Popen:
    """
    Fallback: mount localhost:/tmp via sshfs -f (foreground daemon).
    Requires SSH key to localhost; may fail if keys aren't set up.
    """
    print(
        f"[cgroup_wedge] mounting localhost:/tmp via sshfs -f {mount_point} ...",
        flush=True,
    )
    daemon = subprocess.Popen(
        [
            "sshfs",
            "-f",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "reconnect",
            "localhost:/tmp",
            mount_point,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.5)
    _wait_for_mount(mount_point, timeout=15.0)
    return daemon


def mount_bindfs(source: str, mount_point: str) -> subprocess.Popen:
    """
    Fallback: bind-mount source via bindfs -f (foreground).
    """
    print(
        f"[cgroup_wedge] mounting {source} via bindfs -f {mount_point} ...", flush=True
    )
    daemon = subprocess.Popen(
        ["bindfs", "-f", source, mount_point],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    _wait_for_mount(mount_point, timeout=15.0)
    return daemon


# ---------------------------------------------------------------------------
# D-state wedge logic
# ---------------------------------------------------------------------------


def _find_payload_path(mount_point: str) -> str:
    """Find the first regular file under mount_point."""
    for root, _dirs, files in os.walk(mount_point):
        for f in files:
            return os.path.join(root, f)
    # Generic fallback: read the mount dir itself
    return mount_point


def start_reader(mount_point: str) -> subprocess.Popen:
    """
    Launch a subprocess that reads continuously from the FUSE mount.
    `dd` with a large block size forces real data-block round-trips to the daemon
    (metadata-only operations like `ls` can be served from cache without blocking).
    We intentionally open WITH cached path (no O_DIRECT) — the kernel will still
    call through to FUSE for the actual data pages.
    """
    payload = _find_payload_path(mount_point)
    print(
        f"[cgroup_wedge] starting reader: dd if={payload} of=/dev/null bs=1M",
        flush=True,
    )
    reader = subprocess.Popen(
        ["dd", f"if={payload}", "of=/dev/null", "bs=1M"],
        stderr=subprocess.DEVNULL,
    )
    return reader


def wedge(mount_point: str, fuse_tool: str) -> None:
    """
    Core wedge sequence:
      1. Mount FUSE filesystem (daemon running in foreground as child process).
      2. Start the reader subprocess — daemon is alive, read proceeds initially.
      3. Wait ~500 ms for the reader to have an in-flight read request dispatched
         to the daemon (mid-read freeze maximises D-state probability).
      4. SIGSTOP the daemon — freezes it while it holds the pending response.
      5. Reader's kernel wait is now uninterruptible (D-state).
      6. Sleep forever. SLURM --time fires. Node drains.
    """
    work_dir = tempfile.mkdtemp(prefix="cgroup_wedge_work_")
    print(f"[cgroup_wedge] work dir: {work_dir}", flush=True)

    # --- Mount ---
    if fuse_tool == "squashfuse":
        daemon = mount_squashfuse(work_dir, mount_point)
    elif fuse_tool == "sshfs":
        daemon = mount_sshfs(mount_point)
    else:
        daemon = mount_bindfs("/tmp", mount_point)

    daemon_pid = daemon.pid
    print(
        f"[cgroup_wedge] FUSE daemon PID: {daemon_pid} (tool={fuse_tool})", flush=True
    )

    # Confirm daemon is alive
    if daemon.poll() is not None:
        raise RuntimeError(
            f"FUSE daemon exited immediately (rc={daemon.returncode}). "
            "Check squashfuse/sshfs/bindfs stderr."
        )

    # --- Start reader while daemon is RUNNING (not yet frozen) ---
    reader = start_reader(mount_point)
    print(f"[cgroup_wedge] Reader PID: {reader.pid}", flush=True)

    # Wait ~500 ms so the reader has at least one read request in-flight with
    # the daemon.  Mid-read freeze → uninterruptible wait.
    print("[cgroup_wedge] waiting 500 ms for in-flight read request ...", flush=True)
    time.sleep(0.5)

    # --- SIGSTOP the daemon (freeze mid-response) ---
    print(
        f"[cgroup_wedge] SIGSTOP → daemon {daemon_pid} (freezing FUSE daemon mid-response)",
        flush=True,
    )
    os.kill(daemon_pid, signal.SIGSTOP)
    print(
        f"[cgroup_wedge] daemon {daemon_pid} SIGSTOP'd — reader {reader.pid} should be entering D-state",
        flush=True,
    )

    # Brief pause to let reader enter D-state (kernel round-trip times out)
    time.sleep(1.0)

    # --- Confirm reader is alive (expected: it will block, not exit) ---
    if reader.poll() is not None:
        print(
            f"[cgroup_wedge] WARNING: reader exited already (rc={reader.returncode}). "
            "D-state may not have been achieved. Check observer logs.",
            flush=True,
        )
    else:
        print(
            f"[cgroup_wedge] WEDGE ACTIVE — reader {reader.pid} in D-state, "
            f"FUSE daemon {daemon_pid} SIGSTOP'd",
            flush=True,
        )

    print(
        "[cgroup_wedge] Sleeping forever — SLURM --time will fire in ~2-3 min; "
        "it WILL drain the node.",
        flush=True,
    )
    print(
        f"[cgroup_wedge] >>> D-STATE READER PID: {reader.pid} — check /proc/{reader.pid}/wchan <<<",
        flush=True,
    )
    print(
        f"[cgroup_wedge] >>> D-STATE READER PID: {reader.pid} "
        f"wchan target: fuse_simple_request or fuse_dev_do_read <<<",
        flush=True,
    )

    # Sleep forever — the D-state reader is unkillable
    try:
        signal.pause()
    except Exception:
        time.sleep(int(1e9))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pid = os.getpid()
    mount_point = f"/tmp/cgroup_wedge_fuse_mount_{pid}"

    print("=" * 72, flush=True)
    print("⚠️  DELIBERATE DRAIN TEST — CPU-only cgroup wedge", flush=True)
    print(
        "⚠️  WARNING: No STOP-FILE. No SIGUSR1 trap. This IS the drain test.", flush=True
    )
    print(
        "⚠️  Node WILL drain when SLURM --time fires. Admin must be present.", flush=True
    )
    print("=" * 72, flush=True)

    print(f"[cgroup_wedge] PID: {pid}", flush=True)
    print(f"[cgroup_wedge] Mount point: {mount_point}", flush=True)

    # Check fusermount availability (fuse kernel module)
    fuse_ok = False
    for fm in ("fusermount3", "fusermount"):
        r = subprocess.run([fm, "--version"], capture_output=True)
        if r.returncode == 0:
            ver = r.stdout.decode().strip() or r.stderr.decode().strip()
            print(f"[cgroup_wedge] {fm}: {ver}", flush=True)
            fuse_ok = True
            break
    if not fuse_ok:
        print(
            "FATAL: fusermount / fusermount3 not available — FUSE kernel module may not be loaded."
        )
        print("  Node was NOT drained — no admin action needed.")
        sys.exit(1)

    # Detect FUSE tool
    try:
        fuse_tool = detect_fuse_tool()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        sys.exit(1)

    print(f"[cgroup_wedge] FUSE tool: {fuse_tool}", flush=True)

    # Create mount point
    os.makedirs(mount_point, exist_ok=True)

    try:
        wedge(mount_point, fuse_tool)
    except Exception as exc:
        print(f"[cgroup_wedge] ERROR during wedge setup: {exc}", flush=True)
        # Attempt cleanup so the node is not left with a zombie mount
        try:
            subprocess.run(["fusermount3", "-u", mount_point], capture_output=True)
        except Exception:
            pass
        try:
            subprocess.run(["fusermount", "-u", mount_point], capture_output=True)
        except Exception:
            pass
        print("  Node was NOT drained — no admin action needed.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
