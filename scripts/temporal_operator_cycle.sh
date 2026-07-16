#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --job-name=temporal-op-cycle
#SBATCH --output=_temporal_operator_cycle_%j.log
# Full temporal-operator train/gate cycle with the topology-aware training
# signal: unit tests -> residual-arm sweep (leash x ridge x terminator
# weight, drift-aware composite selection) -> G2 gate eval -> p'/FF' split
# score -> direct-DOF ablation arm -> its gate eval.  Sentinels per stage.
set -u
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR" || export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=12
source .venv/bin/activate
ART=imas_ambix/latent/artifacts/temporal_operator
LABELS=/work/projects/imas_gpu/mast/spine_labels

echo "STAGE_TESTS_START"
OMP_NUM_THREADS=1 python -m pytest tests/latent/test_profile_greens_decoder.py \
  tests/latent/test_temporal_operator.py tests/latent/test_topology_objectives.py \
  -q -m "not slow" --tb=line 2>&1 | tail -3
TESTS_EXIT=${PIPESTATUS[0]}
echo "TESTS_EXIT=$TESTS_EXIT"
[ "$TESTS_EXIT" -ne 0 ] && { echo "PIPELINE_FAILED=tests"; exit 1; }

echo "STAGE_TRAIN_START (residual arm, topology terms, composite selection)"
python scripts/train_temporal_operator.py \
  --labels-dir "$LABELS" \
  --device cuda \
  --init-checkpoint "$ART/temporal_operator_synthetic.pt" \
  --leash-sweep 3.0,10.0,30.0 --eddy-ridge-sweep 0.3,3.0 \
  --terminator-weight-sweep 1.0,10.0 --integrity-weight 1.0 \
  --epochs 120
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
[ $TRAIN_EXIT -ne 0 ] && { echo "PIPELINE_FAILED=training"; exit 1; }

echo "STAGE_GATE_START"
OMP_NUM_THREADS=1 python scripts/temporal_operator_gate_eval.py --workers 8
echo "GATE_EXIT=$?"

echo "STAGE_SPLIT_START"
OMP_NUM_THREADS=1 python scripts/synthetic_eddy_pretrain.py --mode split --workers 8
echo "SPLIT_EXIT=$?"

echo "STAGE_DIRECT_ARM_START (direct-DOF ablation, single sweep point)"
python scripts/train_temporal_operator.py \
  --labels-dir "$LABELS" \
  --device cuda \
  --init-checkpoint "$ART/temporal_operator_synthetic.pt" \
  --leash-sweep 10.0 --eddy-ridge-sweep 0.3 \
  --terminator-weight-sweep 1.0 --integrity-weight 1.0 \
  --arm direct --out-suffix directdof \
  --epochs 120
DIRECT_EXIT=$?
echo "DIRECT_EXIT=$DIRECT_EXIT"
if [ $DIRECT_EXIT -eq 0 ]; then
  echo "STAGE_DIRECT_GATE_START"
  OMP_NUM_THREADS=1 python scripts/temporal_operator_gate_eval.py --workers 8 \
    --checkpoint "$ART/temporal_operator-directdof.pt" --out-suffix directdof
  echo "DIRECT_GATE_EXIT=$?"
fi

echo "PIPELINE_DONE"
