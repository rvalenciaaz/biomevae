#!/usr/bin/env bash
#SBATCH -p ei-gpu
#SBATCH --job-name=single-study-pipeline
#SBATCH --output=single_study_pipeline_%j.out
#SBATCH --error=single_study_pipeline_%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ═══════════════════════════════════════════════════════════════════════════
# single_study_pipeline.slurm – SLURM wrapper for the single-study analysis
#                               pipeline
#
# Runs train -> postprocess -> classify -> figures on the HPC.
# Data must already be extracted using the extract-microbiome-data package
# (https://github.com/rvazdev-ex/extract-microbiome-data) and placed under
# DATA_DIR as sgb_table.tsv, phyla.tsv, sample_metadata.tsv.
#
# Usage:
#   STUDY_NAME=LiJ_2017 \
#   DATA_DIR=/scratch/lij_2017/data \
#   OUTDIR=/scratch/lij_2017 \
#     sbatch single_study_pipeline.slurm
#
#   # Or, chain after training jobs:
#   STUDY_NAME=LiJ_2017 \
#   DATA_DIR=/scratch/lij_2017/data \
#   OUTDIR=/scratch/lij_2017 \
#   SKIP_TRAIN=true \
#     sbatch --dependency=afterok:12345:12346 single_study_pipeline.slurm
#
# Environment variables (set before submission or via --export):
#   OUTDIR            – Root output directory (required)
#   DATA_DIR          – Pre-extracted data directory (required)
#   STUDY_NAME        – Study identifier for job naming / figure titles
#                       (default: "study")
#   LABEL             – Metadata column to classify on (default: "disease")
#   EXTRA_ARGS        – Extra arguments for model training
#                       (default: "--epochs 100 --optuna --optuna-trials 100")
#   SKIP_TRAIN        – Set to "true" to skip training (default: false)
#   SKIP_POSTPROCESS  – Set to "true" to skip postprocessing (default: false)
#   SKIP_CLASSIFY     – Set to "true" to skip classification (default: false)
#   SKIP_AGGREGATE    – Set to "true" to skip aggregation (default: false)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Use SLURM_SUBMIT_DIR because SLURM copies batch scripts to its spool
# directory, making BASH_SOURCE[0] point to the wrong location.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTDIR="${OUTDIR:?Error: OUTDIR must be set}"
DATA_DIR="${DATA_DIR:?Error: DATA_DIR must be set}"
STUDY_NAME="${STUDY_NAME:-study}"
LABEL="${LABEL:-disease}"
EXTRA_ARGS="${EXTRA_ARGS:---epochs 100 --optuna --optuna-trials 100}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_POSTPROCESS="${SKIP_POSTPROCESS:-false}"
SKIP_CLASSIFY="${SKIP_CLASSIFY:-false}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-false}"

PIPELINE_SCRIPT="${SCRIPT_DIR}/single_study_pipeline.sh"
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  # Fallback when sbatch was invoked from the repo root.
  if [[ -f "${SCRIPT_DIR}/hpc/single_study_pipeline.sh" ]]; then
    PIPELINE_SCRIPT="${SCRIPT_DIR}/hpc/single_study_pipeline.sh"
  fi
fi

PIPELINE_ARGS=(
  "${PIPELINE_SCRIPT}"
  --study-name "${STUDY_NAME}"
  --outdir "${OUTDIR}"
  --data-dir "${DATA_DIR}"
  --label "${LABEL}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  PIPELINE_ARGS+=(--extra-args "${EXTRA_ARGS}")
fi

if [[ "${SKIP_TRAIN}" == "true" ]]; then
  PIPELINE_ARGS+=(--skip-train)
fi

if [[ "${SKIP_POSTPROCESS}" == "true" ]]; then
  PIPELINE_ARGS+=(--skip-postprocess)
fi

if [[ "${SKIP_CLASSIFY}" == "true" ]]; then
  PIPELINE_ARGS+=(--skip-classify)
fi

if [[ "${SKIP_AGGREGATE}" == "true" ]]; then
  PIPELINE_ARGS+=(--skip-aggregate)
fi

echo "Running single-study pipeline: ${PIPELINE_ARGS[*]}"
srun --gres=gpu:1 --cpus-per-task=20 --mem=128G bash "${PIPELINE_ARGS[@]}"
