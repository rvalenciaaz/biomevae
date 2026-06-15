#!/usr/bin/env bash
#SBATCH -p ei-short
#SBATCH --job-name=biomevae-agg
#SBATCH --output=biomevae_agg_%j.out
#SBATCH --error=biomevae_agg_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Resolve the aggregate script next to this .slurm file by default.
# Use SLURM_SUBMIT_DIR (the directory where sbatch was invoked) because SLURM
# copies batch scripts to /var/spool/slurmd/job<ID>/, making BASH_SOURCE[0]
# resolve to the spool directory instead of the original location.
#
# When sbatch is invoked from the repo root (e.g. `sbatch hpc/aggregate_results.slurm`),
# SLURM_SUBMIT_DIR points to the repo root, not the hpc/ subdirectory where the
# aggregate script lives.  We check both locations.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ -z "${AGG_SCRIPT:-}" ]]; then
  if [[ -f "${SCRIPT_DIR}/aggregate_results.sh" ]]; then
    AGG_SCRIPT="${SCRIPT_DIR}/aggregate_results.sh"
  elif [[ -f "${SCRIPT_DIR}/hpc/aggregate_results.sh" ]]; then
    AGG_SCRIPT="${SCRIPT_DIR}/hpc/aggregate_results.sh"
  fi
fi

if [[ ! -f "${AGG_SCRIPT:-}" ]]; then
  echo "Error: cannot find aggregate script." >&2
  echo "Searched: ${SCRIPT_DIR}/aggregate_results.sh" >&2
  echo "         ${SCRIPT_DIR}/hpc/aggregate_results.sh" >&2
  echo "SCRIPT_DIR=${SCRIPT_DIR}" >&2
  exit 1
fi

srun --cpus-per-task=4 --mem=16G \
  bash "${AGG_SCRIPT}" \
  "${OUTDIR}" \
  "${INPUT}" \
  "${TAXONOMY}"
