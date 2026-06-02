#!/bin/bash
# drain_study_loop.sh OBS_JOBID
# Rapid kill->relaunch loop of the REAL finetune pipeline to accumulate GPU
# residue until a job wedges at init/teardown and drains the node (the
# single-user cascade). Each cycle: submit real finetune -> it trains -> SLURM
# --time(3min) SIGTERM kills it -> the moment GPUs free, relaunch. No scancel.
set -uo pipefail
OBS_JOB="${1:?need observer jobid}"
NODE=98dci4-gpu-0003
SB=/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/drain_study_finetune.sbatch
CSV=/work/projects/imas_gpu/logs/drain-observer-${OBS_JOB}.csv
CYCLES="${CYCLES:-6}"

drained() { scontrol show node "$NODE" 2>&1 | grep -qiE "State=[^ ]*DRAIN"; }
nstate()  { scontrol show node "$NODE" 2>&1 | grep -oE "State=[^ ]+"; }

for cyc in $(seq 1 "$CYCLES"); do
  if drained; then echo ">>> DRAIN detected before cycle $cyc <<<"; break; fi
  J=$(sbatch --parsable "$SB")
  echo "[cyc $cyc $(date +%H:%M:%S)] node=$(nstate) submitted finetune job=$J"
  # wait until training starts or the job ends
  for w in $(seq 1 60); do
    grep -q "step " "/work/projects/imas_gpu/logs/drain-ft-${J}.out" 2>/dev/null && { echo "  [$(date +%H:%M:%S)] training started"; break; }
    squeue -j "$J" -h 2>/dev/null | grep -q . || { echo "  [$(date +%H:%M:%S)] job ended before training (init wedge?)"; break; }
    drained && { echo "  >>> DRAIN during startup of $J <<<"; break; }
    sleep 4
  done
  # wait for the job to be killed by its --time and release the GPUs
  until ! squeue -j "$J" -h 2>/dev/null | grep -q .; do
    drained && { echo "  >>> DRAIN while $J running <<<"; break; }
    sleep 6
  done
  echo "  [$(date +%H:%M:%S)] job $J gone; node=$(nstate)"
  if drained; then echo ">>> DRAIN after cycle $cyc (job $J) <<<"; break; fi
done

echo "=== loop done @ $(date +%H:%M:%S) node=$(scontrol show node "$NODE" 2>&1 | grep -oE 'State=[^ ]+|Reason=[^|]*' | tr '\n' ' ') ==="
echo "=== D-state wchan histogram (whole study) ==="
grep ",D," "$CSV" 2>/dev/null | awk -F, '{print $4}' | sort | uniq -c | sort -rn | head
