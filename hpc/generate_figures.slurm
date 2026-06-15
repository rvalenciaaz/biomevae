#!/usr/bin/env bash
#SBATCH -p ei-short
#SBATCH --job-name=biomevae-fig
#SBATCH --output=biomevae_fig_%j.out
#SBATCH --error=biomevae_fig_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

FIG_SCRIPT="${FIG_SCRIPT:-${SLURM_SUBMIT_DIR}/generate_figures.sh}"

if [[ ! -f "${FIG_SCRIPT}" ]]; then
  echo "Error: cannot find figure script: ${FIG_SCRIPT}" >&2
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
  exit 1
fi

srun --cpus-per-task=4 --mem=16G \
  bash "${FIG_SCRIPT}" \
  "${MODELS_DIR}" \
  "${METADATA}" \
  "${FIGURES_DIR}" \
  "${INPUT:-}" \
  "${LABEL:-disease}" \
  "${STUDY_NAME:-Study}"
