#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_snakemake.slurm – SLURM driver for the biomevae Snakemake workflow
#
# Submits a long-lived "driver" job that runs ``snakemake`` inside the
# ``biomevae`` mamba environment.  The driver itself uses barely any
# resources – all the real compute happens in the per-rule jobs that
# the SLURM profile (``workflow/profiles/slurm/``) submits to the
# appropriate partitions (ei-gpu for train/postprocess, ei-short for
# classify/figures/aggregate).
#
# The driver runs on ``ei-medium`` because it has to outlive the
# longest rule jobs (training asks for 96h on ei-gpu); ``ei-short``
# would cap it at 2h and SLURM would park it with
# ``(PartitionTimeLimit)``.
#
# Usage
# -----
# Single-study:
#     sbatch hpc/run_snakemake.slurm \
#         -s workflow/Snakefile \
#         --configfile workflow/config/single_study.yaml
#
# Meta (all multi-label studies):
#     sbatch hpc/run_snakemake.slurm \
#         -s workflow/Snakefile.meta \
#         --configfile workflow/config/meta_multi_label.yaml
#
# Config overrides:
#     sbatch hpc/run_snakemake.slurm \
#         -s workflow/Snakefile \
#         --config data_root=/scratch/data output_root=/scratch/runs \
#                  study=LiJ_2017 label=disease
#
# Re-run a single target:
#     sbatch hpc/run_snakemake.slurm \
#         -s workflow/Snakefile \
#         --configfile workflow/config/single_study.yaml \
#         /scratch/runs/LiJ_2017/models/tax-vae/classify/classification_results.json
#
# Environment variables (override with ``sbatch --export=...``):
#     MAMBA_EXEC        Path to mamba binary
#                       (default: /hpc-home/her24bip/miniconda3/condabin/mamba)
#     MAMBA_ROOT_PREFIX MAMBA_ROOT_PREFIX for `mamba shell hook`
#                       (default: /hpc-home/her24bip/.local/share/mamba)
#     CONDA_ENV         Environment to activate (default: biomevae)
#     SMK_PROFILE       Snakemake profile to use
#                       (default: workflow/profiles/slurm)
# ═══════════════════════════════════════════════════════════════════════════
#SBATCH -p ei-medium
#SBATCH --job-name=biomevae-smk
#SBATCH --output=biomevae_smk_%j.out
#SBATCH --error=biomevae_smk_%j.err
#SBATCH --time=216:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

# ── Locate the repo root ────────────────────────────────────────────────
# SLURM copies batch scripts to its spool directory, so BASH_SOURCE
# would resolve to the wrong location.  Use SLURM_SUBMIT_DIR instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -d "${SCRIPT_DIR}/workflow" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/../workflow" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "Error: cannot find workflow/ relative to SLURM_SUBMIT_DIR=${SCRIPT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
echo "Working directory: ${REPO_ROOT}"

# ── Activate the biomevae conda/mamba environment ──────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

if [[ ! -x "${MAMBA_EXEC}" ]]; then
  echo "Error: mamba executable not found at ${MAMBA_EXEC}" >&2
  echo "Set MAMBA_EXEC (and optionally MAMBA_ROOT_PREFIX / CONDA_ENV)" >&2
  exit 1
fi

# ``mamba shell hook`` prints the shell code needed to enable
# ``mamba activate`` inside a non-interactive batch shell.
eval "$("${MAMBA_EXEC}" shell hook --shell bash)"
mamba activate "${CONDA_ENV}"

echo "Active environment: ${CONDA_DEFAULT_ENV:-<none>}"
echo "snakemake: $(command -v snakemake || echo '<not found>')"

# ── Pre-create directories the SLURM profile relies on ─────────────────
mkdir -p logs/slurm

# ── Launch Snakemake in cluster mode ───────────────────────────────────
SMK_PROFILE="${SMK_PROFILE:-workflow/profiles/slurm}"

exec snakemake --profile "${SMK_PROFILE}" "$@"
