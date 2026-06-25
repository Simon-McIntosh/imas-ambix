"""SSH tunnel management for reaching a served model from a login node.

A model served on a SLURM GPU compute node binds its OpenAI-compatible
port (default 18800) on that node.  Login nodes cannot reach the compute
node's port directly, so a client (or the ``status`` readiness probe)
needs a local forward: ``localhost:18800 → <compute-node>:18800``.

This module establishes and tracks that forward.  It mirrors the proven
pattern used by ``imas_codex.remote.tunnel`` (PID-file tracking,
``autossh`` when available with a plain ``ssh -f -N`` fallback, keepalive
tuning) but is scoped to the ambix serve port and discovers the compute
node from the running serve job rather than a fixed service name.

Same-port forwarding is used (local port == remote port) so the client
URL is simply ``http://localhost:<port>``.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# PID files tracking our tunnel processes, keyed by compute-node host.
_PID_DIR = Path.home() / ".local" / "share" / "ambix" / "tunnels"

# SSH keepalive — 15 × 3 = 45 s drop detection (matches imas-codex).
_SSH_ALIVE_INTERVAL = 15
_SSH_ALIVE_COUNT_MAX = 3

# Common SSH options for tunnel connections.  We tolerate failures of
# unrelated forwards from the user's ssh config (ExitOnForwardFailure=no)
# and never share a ControlMaster (so stopping our tunnel can't disturb
# an interactive session).
SSH_TUNNEL_OPTS: list[str] = [
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "TCPKeepAlive=yes",
    "-o", f"ServerAliveInterval={_SSH_ALIVE_INTERVAL}",
    "-o", f"ServerAliveCountMax={_SSH_ALIVE_COUNT_MAX}",
    "-o", "ConnectTimeout=10",
    "-o", "ExitOnForwardFailure=yes",
]  # fmt: skip


def _pid_file(host: str) -> Path:
    return _PID_DIR / f"{host}.pid"


def _write_pid(host: str, pid: int) -> None:
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    _pid_file(host).write_text(str(pid))


def _read_pid(host: str) -> int | None:
    """Return the recorded tunnel PID for *host* if still alive, else None."""
    path = _pid_file(host)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        path.unlink(missing_ok=True)
        return None


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if something is listening on *host*:*port*."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def is_port_bound_by_ssh(port: int) -> bool:
    """Return True if *port* is bound specifically by an ssh process."""
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    for line in result.stdout.splitlines():
        if f":{port}" in line and "ssh" in line.lower():
            return True
    return False


def tunnel_status(port: int) -> str:
    """Classify the local tunnel port.

    Returns ``"up"`` (bound by ssh), ``"foreign"`` (bound by something
    that is not ssh — e.g. a local server), or ``"down"`` (nothing
    listening).
    """
    if not is_port_open(port):
        return "down"
    return "up" if is_port_bound_by_ssh(port) else "foreign"


def discover_serving_node(job_name: str) -> str | None:
    """Return the compute node a RUNNING serve job is on, via local squeue.

    *job_name* is the profile slug (serve jobs are named after the slug).
    Returns the node hostname, or None when no RUNNING job matches.
    """
    try:
        result = subprocess.run(
            ["squeue", "-h", "-n", job_name, "-u",
             os.environ.get("USER", ""), "-t", "R", "--format=%N"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    node = result.stdout.strip().splitlines()
    return node[0].strip() if node and node[0].strip() else None


def start_tunnel(host: str, port: int, timeout: float = 15.0) -> bool:
    """Establish ``localhost:port → host:port`` if not already up.

    Prefers ``autossh`` for auto-reconnection; falls back to ``ssh -f -N``.
    Returns True if the tunnel is active (pre-existing or newly created).
    """
    state = tunnel_status(port)
    if state == "up":
        return True
    if state == "foreign":
        logger.warning("Port %d already bound by a non-ssh process", port)
        return False

    use_autossh = shutil.which("autossh") is not None
    forward = f"{port}:127.0.0.1:{port}"
    if use_autossh:
        cmd = ["autossh", "-M", "0", "-f", "-N", *SSH_TUNNEL_OPTS,
               "-L", forward, host]
        env = {**os.environ, "AUTOSSH_GATETIME": "0", "AUTOSSH_POLL": "30"}
    else:
        cmd = ["ssh", "-f", "-N", *SSH_TUNNEL_OPTS, "-L", forward, host]
        env = None

    try:
        subprocess.run(
            cmd, timeout=timeout, check=True,
            capture_output=True, text=True, env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Tunnel start failed: %s", exc.stderr.strip() if exc.stderr else exc
        )
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Tunnel start failed: %s", exc)
        return False

    # ssh -f forks once the forward is bound (ExitOnForwardFailure=yes).
    time.sleep(0.3)
    _record_pid(host, port)
    for _ in range(8):
        if is_port_open(port):
            return True
        time.sleep(0.5)
    return False


def _record_pid(host: str, port: int) -> None:
    """Find the ssh/autossh process for our forward and record its PID."""
    for prog in ("autossh", "ssh"):
        result = subprocess.run(
            ["pgrep", "-f", f"{prog}.*-L {port}:.*{host}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for pid_str in result.stdout.strip().splitlines():
                try:
                    _write_pid(host, int(pid_str))
                    return
                except ValueError:
                    continue


def stop_tunnel(host: str, port: int) -> bool:
    """Stop the tunnel to *host*.  Returns True if a process was killed.

    Targets the recorded PID first (killing its process group to catch an
    autossh+ssh pair), then falls back to matching our exact forward
    pattern so an orphaned tunnel is still reaped without touching the
    user's unrelated ssh sessions.
    """
    stopped = False
    pid = _read_pid(host)
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            stopped = True
        except (ProcessLookupError, PermissionError):
            pass
        _pid_file(host).unlink(missing_ok=True)

    result = subprocess.run(
        ["pgrep", "-f", f"-L {port}:.*{host}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for pid_str in result.stdout.strip().splitlines():
            try:
                os.kill(int(pid_str), signal.SIGTERM)
                stopped = True
            except (ProcessLookupError, PermissionError, ValueError):
                pass
    return stopped


__all__ = [
    "SSH_TUNNEL_OPTS",
    "discover_serving_node",
    "is_port_bound_by_ssh",
    "is_port_open",
    "start_tunnel",
    "stop_tunnel",
    "tunnel_status",
]
