#!/usr/bin/env bash
#SBATCH --job-name=nova-playable
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/nova-playable-%j.log
#SBATCH --error=/work/projects/imas_gpu/logs/nova-playable-%j.log

set -euo pipefail

readonly NOVA_ROOT=/home/ITER/mcintos/Code/nova
readonly AMBIX_ROOT=/home/ITER/mcintos/Code/imas-ambix
readonly PYTHON=/home/ITER/mcintos/Code/imas-ambix/.venv/bin/python
readonly APP="${NOVA_ROOT}/apps/playable"
readonly PORT=18506
readonly DECODER=imas_ambix.worldmodel.flux_conditioned_decoder:FluxConditionedDecoder
readonly DEFAULT_CONFIG=/work/projects/imas_gpu/ambix/flux_decoder/latest/runtime.json

decoder_config=${IMAS_AMBIX_FLUX_DECODER:-${DEFAULT_CONFIG}}
dry_run=false
while (($#)); do
  case "$1" in
    --decoder-config)
      decoder_config=$2
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

readonly decoder_config
readonly allocation_log="${PLAYABLE_ALLOCATION_LOG:-${HOME}/.local/share/nova/playable/allocation.log}"

required=("${NOVA_ROOT}" "${AMBIX_ROOT}" "${PYTHON}" "${APP}")
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing required path: %s\n' "${path}" >&2
    exit 2
  fi
done

printf 'NOVA_ROOT=%s\n' "${NOVA_ROOT}"
printf 'AMBIX_ROOT=%s\n' "${AMBIX_ROOT}"
printf 'PLAYABLE_PYTHON=%s\n' "${PYTHON}"
printf 'PLAYABLE_APP=%s\n' "${APP}"
printf 'PLAYABLE_PORT=%s\n' "${PORT}"
printf 'PLAYABLE_DECODER=%s\n' "${DECODER}"
printf 'IMAS_AMBIX_FLUX_DECODER=%s\n' "$(realpath -m -- "${decoder_config}")"
printf 'PLAYABLE_ALLOCATION_LOG=%s\n' "$(realpath -m -- "${allocation_log}")"

if [[ "${dry_run}" == true ]]; then
  printf 'DRY_RUN_EXIT_STATUS=0\n'
  exit 0
fi
if [[ ! -f "${decoder_config}" ]]; then
  printf 'missing decoder runtime configuration: %s\n' "${decoder_config}" >&2
  exit 2
fi

export TMPDIR=/tmp
export PYTHONPATH="${AMBIX_ROOT}:${NOVA_ROOT}"
export IMAS_AMBIX_FLUX_DECODER="${decoder_config}"
export JAX_PLATFORMS=cuda,cpu

node=$(hostname -s)
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
revision=$(git -C "${NOVA_ROOT}" rev-parse HEAD)
printf 'AMBIX_SOURCE_REVISION=%s\n' "$(git -C "${AMBIX_ROOT}" rev-parse HEAD)"
mkdir -p "$(dirname "${allocation_log}")"
printf '%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_JOB_ID:-unknown}" "${node}" "${PORT}" "${started}" "${revision}" \
  >"${allocation_log}"
printf 'ALLOCATION_LOG=%s\n' "${allocation_log}"

exec "${PYTHON}" -m bokeh serve "${APP}" \
  --address 0.0.0.0 \
  --port "${PORT}" \
  --allow-websocket-origin "localhost:${PORT}" \
  --allow-websocket-origin "127.0.0.1:${PORT}" \
  --allow-websocket-origin "${node}:${PORT}" \
  --args --decoder "${DECODER}"
