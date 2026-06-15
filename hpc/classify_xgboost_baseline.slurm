#!/usr/bin/env bash
#SBATCH -p ei-short
#SBATCH --job-name=biomevae-clf-baseline
#SBATCH --output=biomevae_clf_baseline_%j.out
#SBATCH --error=biomevae_clf_baseline_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Prefer the explicitly exported path from submit_all_classify.sh
CLF_BASELINE_SCRIPT="${CLF_BASELINE_SCRIPT:-${SLURM_SUBMIT_DIR}/classify_xgboost_baseline.sh}"

if [[ ! -f "${CLF_BASELINE_SCRIPT}" ]]; then
  echo "Error: cannot find baseline classify script: ${CLF_BASELINE_SCRIPT}" >&2
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
  exit 1
fi

srun --cpus-per-task=4 --mem=16G \
  bash "${CLF_BASELINE_SCRIPT}" \
  "${INPUT}" \
  "${METADATA}" \
  "${LABEL}" \
  "${OUTDIR}"
