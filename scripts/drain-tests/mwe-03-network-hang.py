"""
MWE-03: Network Black-Hole D-State Hang
========================================
PURPOSE
  Tests whether a blocking TCP connect() to a DROP-firewalled host puts
  the Python process into uninterruptible sleep (D-state), which would
  make it immune to SIGKILL and cause a node drain when SLURM tries to
  clean up the job.

  This reproduces Event 2 (2026-05-27, job 1208591): a torch download
  attempt to the internet from a firewall-DROP betelgeuse node caused
  D-state, leading to a node drain.

RISK LEVEL
  ⚠️  UNCERTAIN DRAIN — may or may not drain depending on kernel/NIC
  behaviour. If the TCP stack enters D-state waiting for network, the
  node WILL drain. If it uses S-state (interruptible sleep, futex),
  SLURM can kill it cleanly.

  On betelgeuse (firewall DROP, no REJECT): very likely D-state.
  Expected recovery: scontrol resume (no GPU reset needed).
  Keep admin contact ready before running this test.

WHAT IT DOES
  1. On a single process (no DDP needed — network D-state is independent
     of NCCL/GPU).
  2. Prints its PID and hostname for external /proc/<pid>/status monitoring.
  3. Calls socket.connect() to a non-routable host (103.21.244.0:443)
     with no timeout — this blocks indefinitely if the firewall DROPs.
  4. The script never self-terminates. It must be killed externally via
     scancel or kill.
  5. Expected external observation: ``cat /proc/<pid>/status | grep State``
     shows "D (disk sleep)" if the network stack is in D-state.

HOW TO TEST
  From a login node / separate shell:

  1. Submit:
       sbatch mwe-03-network-hang.sbatch
       # or for interactive:
       srun --partition=betelgeuse --reservation=gpu_0003_grpA \\
            --account=grpa --gres=gpu:0 --cpus-per-task=1 \\
            --mem=1G --pty bash -l

  2. Watch the log until "ATTEMPTING TCP CONNECT" appears.

  3. SSH to the compute node and check process state:
       ssh <compute-node>
       cat /proc/<pid>/status | grep -E "^State|^Name"

  4. Issue scancel:
       scancel <jobid>

  5. Check node state after ~60 s (UnkillableStepTimeout):
       sinfo -N -n 98dci4-gpu-0003 --noheader -o "%T"
       # "drain" → node drained (confirm mechanism)
       # "idle" or "allocated" → no drain (S-state, SIGKILL worked)

ENVIRONMENT VARIABLES
  TARGET_HOST   IP to connect to (default: 103.21.244.0 — non-routable)
  TARGET_PORT   Port (default: 443)
  WAIT_BEFORE   Seconds to wait before connect, giving time to observe
                PID (default: 10)
  DRAIN_TEST_LOG_DIR  Log directory (default: /tmp)
"""

import os
import socket
import sys
import time

TARGET_HOST = os.environ.get("TARGET_HOST", "103.21.244.0")
TARGET_PORT = int(os.environ.get("TARGET_PORT", "443"))
WAIT_BEFORE = float(os.environ.get("WAIT_BEFORE", "10"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

import socket as _socket_module

hostname = _socket_module.gethostname()
pid = os.getpid()

print(f"[MWE-03] hostname={hostname}  pid={pid}  job={JOB_ID}", flush=True)
print(f"[MWE-03] To monitor process state from compute node:", flush=True)
print(f"[MWE-03]   ssh {hostname}", flush=True)
print(f"[MWE-03]   cat /proc/{pid}/status | grep -E '^State|^Name'", flush=True)
print(f"[MWE-03]   # D = disk sleep (uninterruptible) → drain risk", flush=True)
print(f"[MWE-03]   # S = sleeping (interruptible) → safe to kill", flush=True)
print(f"", flush=True)
print(f"[MWE-03] Waiting {WAIT_BEFORE:.0f} s before connect (time to note PID)...", flush=True)

time.sleep(WAIT_BEFORE)

print(f"[MWE-03] ATTEMPTING TCP CONNECT to {TARGET_HOST}:{TARGET_PORT} (no timeout)", flush=True)
print(f"[MWE-03] This will block forever if the firewall DROPs packets.", flush=True)
print(f"[MWE-03] Issue 'scancel {JOB_ID}' now, then check node state in 70 s.", flush=True)

t_start = time.monotonic()

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TARGET_HOST, TARGET_PORT))
    elapsed = time.monotonic() - t_start
    print(f"[MWE-03] UNEXPECTED: connect() returned after {elapsed:.2f} s", flush=True)
    print(f"[MWE-03] Host is reachable — network hang test not valid on this node.", flush=True)
    sock.close()
    sys.exit(0)

except OSError as exc:
    elapsed = time.monotonic() - t_start
    print(f"[MWE-03] connect() raised after {elapsed:.2f} s: {exc}", flush=True)
    print(f"[MWE-03] REJECT received (not DROP) — process stayed killable, no drain risk.", flush=True)
    sys.exit(0)

except KeyboardInterrupt:
    elapsed = time.monotonic() - t_start
    print(f"[MWE-03] KeyboardInterrupt after {elapsed:.2f} s", flush=True)
    sys.exit(0)

# If we reach here, connect() returned successfully — unusual
print("[MWE-03] Connected (unexpected). Closing.", flush=True)
sock.close()
