#!/usr/bin/env bash
#SBATCH -p ei-gpu
#SBATCH --job-name=biomevae-pp
#SBATCH --output=biomevae_pp_%x_%j.out
#SBATCH --error=biomevae_pp_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

export THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Resolve the postprocess script next to this .slurm file by default.
# Use SLURM_SUBMIT_DIR because SLURM copies batch scripts to its spool
# directory, making BASH_SOURCE[0] point to the wrong location.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PP_SCRIPT="${PP_SCRIPT:-${SCRIPT_DIR}/postprocess_model.sh}"

if [[ ! -f "${PP_SCRIPT}" ]]; then
  echo "Error: cannot find postprocess script: ${PP_SCRIPT}" >&2
  echo "SCRIPT_DIR=${SCRIPT_DIR}" >&2
  exit 1
fi

srun --gres=gpu:1 --cpus-per-task=20 --mem=128G \
  bash "${PP_SCRIPT}" \
  "${MODEL_OUTDIR}" \
  "${INPUT}" \
  "${TAXONOMY}"
