#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_snakemake_loso.slurm – SLURM driver for the biomevae LOSO pipeline
#
# Strict leave-one-study-out cross-cohort generalisation evaluation
# (``workflow/Snakefile.loso``).  Launches the merge → train → embed →
# classify(per-fold) → diagnostic → aggregate DAG; per-rule resources
# are read from each rule's ``resources:`` block via the SLURM profile
# at ``workflow/profiles/slurm/config.yaml``.
#
# Compute envelope (CRC, 11 studies × 5 models × 5 seeds):
#   * 1 ``loso_prepare`` job  (ei-short, 1h, CPU)
#   * 5 ``loso_train_model`` jobs   (ei-gpu, 96h, 1× GPU each)
#   * 5 ``loso_diagnostic`` jobs    (ei-short, 30m, CPU)
#   * 55 ``loso_classify_fold`` jobs (ei-short, 1h, CPU each)
#   * 5 ``loso_aggregate_model`` + 1 ``loso_summary`` (ei-short, 15m)
#
# Minimal invocation:
#
#     sbatch hpc/run_snakemake_loso.slurm
#
# Override config keys on the CLI (extra args go straight to snakemake;
# the default ``-s`` and ``--configfile`` are still injected unless you
# pass your own):
#
#     sbatch hpc/run_snakemake_loso.slurm \
#         --config data_root=/scratch/extracted output_root=/scratch/loso
#
# Switch to a different disease group (e.g. IBD) by passing a custom
# config file:
#
#     sbatch hpc/run_snakemake_loso.slurm \
#         --configfile workflow/config/loso_ibd.yaml
#
# Environment variables (override with ``sbatch --export=...``):
#     MAMBA_EXEC        Path to mamba binary
#                       (default: /hpc-home/her24bip/miniconda3/condabin/mamba)
#     MAMBA_ROOT_PREFIX MAMBA_ROOT_PREFIX for `mamba shell hook`
#                       (default: /hpc-home/her24bip/.local/share/mamba)
#     CONDA_ENV         Environment to activate (default: biomevae)
#     SMK_PROFILE       Snakemake profile (default: workflow/profiles/slurm)
# ═══════════════════════════════════════════════════════════════════════════
#SBATCH -p ei-medium
#SBATCH --job-name=biomevae-smk-loso
#SBATCH --output=biomevae_smk_loso_%j.out
#SBATCH --error=biomevae_smk_loso_%j.err
#SBATCH --time=336:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

# ── locate repo root ──────────────────────────────────────────────────
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -d "${SCRIPT_DIR}/workflow" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/../workflow" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "Error: cannot find workflow/ relative to ${SCRIPT_DIR}" >&2
  exit 1
fi
cd "${REPO_ROOT}"
echo "Working directory: ${REPO_ROOT}"

# ── activate the biomevae environment ────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

if [[ ! -x "${MAMBA_EXEC}" ]]; then
  echo "Error: mamba executable not found at ${MAMBA_EXEC}" >&2
  exit 1
fi
eval "$("${MAMBA_EXEC}" shell hook --shell bash)"
mamba activate "${CONDA_ENV}"
echo "Active environment: ${CONDA_DEFAULT_ENV:-<none>}"
echo "snakemake: $(command -v snakemake || echo '<not found>')"

# ── pre-create directories the SLURM profile relies on ───────────────
mkdir -p logs/slurm

# ── launch Snakemake ─────────────────────────────────────────────────
SMK_PROFILE="${SMK_PROFILE:-workflow/profiles/slurm}"

# Inject the LOSO Snakefile and config unless the user already passed
# their own.  Previously these defaults were only added when ``$#`` was
# zero, so the documented form
#   sbatch hpc/run_snakemake_loso.slurm --config data_root=…
# fell back to the single-study Snakefile and crashed in
# ``resolve_studies()`` with "Config must set one of: 'study'…".
PREPEND_ARGS=()
USER_ARGS=("$@")
USER_HAS_SNAKEFILE=0
USER_HAS_CONFIGFILE=0
for arg in "${USER_ARGS[@]}"; do
  case "${arg}" in
    -s|--snakefile|--snakefile=*) USER_HAS_SNAKEFILE=1 ;;
    --configfile|--configfile=*|--configfiles|--configfiles=*) USER_HAS_CONFIGFILE=1 ;;
  esac
done
if [[ "${USER_HAS_SNAKEFILE}" -eq 0 ]]; then
  PREPEND_ARGS+=(-s workflow/Snakefile.loso)
fi
if [[ "${USER_HAS_CONFIGFILE}" -eq 0 ]]; then
  PREPEND_ARGS+=(--configfile workflow/config/loso_crc.yaml)
fi

exec snakemake --profile "${SMK_PROFILE}" "${PREPEND_ARGS[@]}" "${USER_ARGS[@]}"
