#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_snakemake_meta.slurm – SLURM driver for the biomevae meta pipeline
#
# Same as ``hpc/run_snakemake.slurm`` but tuned for the meta pipeline
# (``workflow/Snakefile.meta``), which loops the full single-study
# workflow across every multi-label case/control study under
# ``data_root``.  A single study's training stage alone asks for 96h on
# ``ei-gpu``; the meta pipeline multiplies that by the number of
# studies × 10 model variants and also has to wait for postprocess,
# classify, figures and aggregate to finish, so the driver needs to
# stay alive for much longer than the 48h the single-study driver gets.
#
# This script defaults to ``-s workflow/Snakefile.meta`` and
# ``--configfile workflow/config/meta_multi_label.yaml`` when no flags
# are passed, so the minimal invocation is just:
#
#     sbatch hpc/run_snakemake_meta.slurm
#
# Override config on the CLI (extra args are forwarded to Snakemake):
#
#     sbatch hpc/run_snakemake_meta.slurm \
#         --config data_root=/scratch/extracted output_root=/scratch/meta
#
# Or point at a different Snakefile / config entirely:
#
#     sbatch hpc/run_snakemake_meta.slurm \
#         -s workflow/Snakefile.meta \
#         --configfile /path/to/custom_meta.yaml
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
#SBATCH --job-name=biomevae-smk-meta
#SBATCH --output=biomevae_smk_meta_%j.out
#SBATCH --error=biomevae_smk_meta_%j.err
#SBATCH --time=336:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

# ── Locate the repo root ────────────────────────────────────────────────
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

eval "$("${MAMBA_EXEC}" shell hook --shell bash)"
mamba activate "${CONDA_ENV}"

echo "Active environment: ${CONDA_DEFAULT_ENV:-<none>}"
echo "snakemake: $(command -v snakemake || echo '<not found>')"

# ── Pre-create directories the SLURM profile relies on ─────────────────
mkdir -p logs/slurm

# ── Launch Snakemake in cluster mode ───────────────────────────────────
SMK_PROFILE="${SMK_PROFILE:-workflow/profiles/slurm}"

# Default to the meta Snakefile + config when the user passes no args.
if [[ "$#" -eq 0 ]]; then
  set -- -s workflow/Snakefile.meta \
         --configfile workflow/config/meta_multi_label.yaml
fi

exec snakemake --profile "${SMK_PROFILE}" "$@"
