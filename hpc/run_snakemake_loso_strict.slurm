#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_snakemake_loso_strict.slurm – SLURM driver for the STRICT LOSO pipeline
#
# Strict leave-one-study-out: re-trains the encoder on N-1 studies for
# every (model, held_out) fold (``workflow/Snakefile.loso_strict``).
# Eliminates the model-side leakage of the non-strict pipeline by
# keeping the held-out cohort out of representation learning AND
# Optuna's val-loss-driven hyperparameter selection.
#
# Compute envelope, CRC group with 11 studies × 8 models × 5 seeds,
# 100 Optuna trials per fold:
#   * 1 ``strict_prepare`` job        (ei-short, 1h, CPU)
#   * 11 ``strict_fold_split`` jobs   (ei-short, 30m, CPU each)
#   * 88 ``strict_train_fold`` jobs   (ei-gpu, 96h, 1× GPU each)
#   * 88 ``strict_encode_holdout`` jobs (ei-gpu, 1h, 1× GPU each)
#   * 88 ``strict_concat_embeddings`` jobs (ei-short, 15m, CPU each)
#   * 88 ``strict_classify_fold`` jobs (ei-short, 1h, CPU each)
#   * 88 ``strict_diagnostic_fold`` jobs (ei-short, 30m, CPU each)
#   * 8 ``strict_aggregate_model`` + 1 ``strict_summary`` (ei-short, 15m)
#
# That is ~N× the cost of ``run_snakemake_loso.slurm`` because Optuna
# now runs once per fold instead of once per model.  The driver job
# itself is light (it just dispatches Snakemake to SLURM); the SLURM
# profile under ``workflow/profiles/slurm/`` reads each rule's
# ``resources:`` block to set partition / runtime / GPU per submitted
# child job.
#
# Minimal invocation:
#
#     sbatch hpc/run_snakemake_loso_strict.slurm
#
# Override config keys on the CLI (matches the non-strict driver's
# convention; extra args go straight to snakemake, the default ``-s``
# and ``--configfile`` are still injected unless you pass your own):
#
#     sbatch hpc/run_snakemake_loso_strict.slurm \
#         --config data_root=/scratch/extracted output_root=/scratch/loso
#
# Switch to a different disease group:
#
#     sbatch hpc/run_snakemake_loso_strict.slurm \
#         --configfile workflow/config/loso_strict_ibd.yaml
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
#SBATCH --job-name=biomevae-smk-loso-strict
#SBATCH --output=biomevae_smk_loso_strict_%j.out
#SBATCH --error=biomevae_smk_loso_strict_%j.err
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

# Inject the strict LOSO Snakefile and config unless the user already
# passed their own.
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
  PREPEND_ARGS+=(-s workflow/Snakefile.loso_strict)
fi
if [[ "${USER_HAS_CONFIGFILE}" -eq 0 ]]; then
  PREPEND_ARGS+=(--configfile workflow/config/loso_strict_crc.yaml)
fi

exec snakemake --profile "${SMK_PROFILE}" "${PREPEND_ARGS[@]}" "${USER_ARGS[@]}"
