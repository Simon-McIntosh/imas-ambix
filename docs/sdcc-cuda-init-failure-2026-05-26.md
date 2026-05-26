# SDCC ticket draft — CUDA init failure on `98dci4-gpu-0003`

**Date:** 2026-05-26 17:30 CEST onward
**Node:** `98dci4-gpu-0003` (betelgeuse partition)
**Reporter:** Simon McIntosh (simon.mcintosh@iter.org)
**Reservation:** `gpu_0003_grpA` (Group A, 4× H200)

## Symptom

Every CUDA-driver call inside a Group A SLURM allocation on
`98dci4-gpu-0003` fails with **`cuInit() rc = 3`**
(`CUDA_ERROR_NOT_INITIALIZED`). Reproducible from pure C with no PyTorch
or other Python framework involved.

`nvidia-smi`, `nvidia-smi -L`, and NVML queries continue to work — the
problem is restricted to the CUDA runtime / driver API path.

## Timeline

| Time (CEST) | Event |
|---|---|
| 17:25 | `smoke6` (job 1207568): 3-shot Open-MAGVIT2 encode on Group A succeeded, 21.6 s wall, GPU 0 OK. |
| 17:31 | Group B job `1207566` (user `murawan`) starts on the same node. |
| 17:31 → now | All subsequent CUDA-init attempts in Group A fail with `cuInit rc=3`. Tested with 5 separate sbatch submissions (jobs 1207573, 1207596, 1207609, 1207627, 1207644). |

## Minimal repro (pure C, no torch)

```bash
sbatch --partition=betelgeuse --reservation=gpu_0003_grpA \
       --account=grpa --gres=gpu:1 --cpus-per-task=1 \
       --mem=4G --time=00:02:00 --wrap='
cat > /tmp/t.c <<EOF
#include <stdio.h>
#include <dlfcn.h>
int main() {
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    int (*cuInit)(unsigned) = (int(*)(unsigned))dlsym(h, "cuInit");
    int (*cnt)(int*) = (int(*)(int*))dlsym(h, "cuDeviceGetCount");
    printf("cuInit rc=%d\n", cuInit(0));
    int n = -1; cnt(&n);
    printf("cuDeviceGetCount n=%d\n", n);
    return 0;
}
EOF
gcc /tmp/t.c -o /tmp/t -ldl && /tmp/t'
```

**Expected:** `cuInit rc=0  cuDeviceGetCount n=1`
**Observed:** `cuInit rc=3  cuDeviceGetCount n=0` *(rc=3 = `CUDA_ERROR_NOT_INITIALIZED`)*

## Environment state at failure

```
Driver:               595.58.03
CUDA (reported):      13.2
GPU 0 (visible):      H200 NVL, UUID GPU-a9aae28e-bf8d-…   ← nvidia-smi sees it fine
Allocated to job:     /dev/nvidia2 (SLURM_STEP_GPUS=2)
Cgroup device perms:  /dev/nvidia2 = OK, /dev/nvidia[0,1,3-7] = EPERM (expected)
/dev/nvidiactl:       openable
NVML query path:      works
CUDA driver API path: fails
```

Both PyTorch builds confirm the same failure:
- `torch 2.1.1+cu118` (Open-MAGVIT2 venv) — `cuda.is_available() = False, device_count() = 1`
- `torch 2.11.0+cu130` (main project venv) — same

`nvidia-smi --query-gpu=…` reports 0 % util, 0 MiB used, no processes — yet `cuInit` refuses.

## Hypothesised cause

Driver/cgroup state degraded after concurrent Group A + Group B activity. Around
17:31 (when Group B job `1207566` started, the *only* concurrent change on
the node), Group A lost CUDA-runtime access while keeping NVML access. Repeat
fresh SLURM allocations have not recovered the state — looks like a node-level
driver lock rather than a per-job cgroup issue.

Possible mechanisms (in decreasing order of confidence):

1. **Driver state machine wedged** between two cgroup contexts on the same
   physical device — driver loaded fine on first Group A use, then Group B's
   init left the driver in a state Group A's subsequent `cuInit` can't enter.
   Would explain why NVML still works (independent path) while CUDA runtime
   does not.
2. **Stale resource lock from one of my earlier failed multi-GPU jobs**
   (1207573, 1207596) — but `lsof /dev/nvidia*` shows no leaked file
   descriptors and `fuser` is empty, so this seems unlikely.
3. **NVRM/UVM kernel module reference count off** — fixable only by
   `nvidia-smi --gpu-reset` (requires root) or a node reboot.

## Proposed fix

In rough order of disruption:

1. **`nvidia-smi --gpu-reset` (root)** on `98dci4-gpu-0003` while no jobs are
   running. Lowest impact if a window is available.
2. **Reload nvidia kernel modules** (`rmmod nvidia_uvm nvidia_modeset nvidia` +
   `modprobe nvidia_uvm`) — clean reset of driver state. Requires draining the
   node first.
3. **Reboot `98dci4-gpu-0003`** if module reload is not feasible. Most
   conservative but takes the node out for the duration.

Once the driver is healthy, please consider:

- **Investigating whether SLURM's gres cgroup plugin is correctly isolating
  CUDA runtime state between concurrent Group A and Group B jobs.** If
  Group B can wedge Group A's CUDA path without crashing the node, the
  cgroup-device isolation may not be covering UVM/NVRM state. NVIDIA's
  CUDA Container Toolkit pattern (mounting `/dev/nvidia<N>` per cgroup) is
  in place but may need a kernel-module config tweak (e.g.
  `NVreg_RegisterPCIDriver=0` or `nvidia-modprobe -c <gpu>` per cgroup).

If the fix can be applied during a low-utilisation window (Group A's M1
training push needs this reservation through end of week), please ping us so
we can drain our queue before the reset.

## Workaround in the meantime

None on this node. Open-MAGVIT2 bulk-encode (the immediate need) is
blocked. Local-only Python work can continue on login nodes.

---

**Diagnostic command for re-verification by SDCC:** the pure-C repro above
takes ~5 s and gives a clear pass/fail signal. We can re-run on demand.

— Simon
