#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# generate_figures.sh – generate publication-quality figures for a
#                       single-study biomevae run
#
# Usage (called by generate_figures.slurm via srun):
#   ./generate_figures.sh <MODELS_DIR> <METADATA> <FIGURES_DIR> \
#                         [INPUT] [LABEL] [STUDY_NAME]
#
# Arguments:
#   MODELS_DIR  – directory containing per-model sub-folders with results
#   METADATA    – path to sample_metadata.tsv
#   FIGURES_DIR – output directory for figures
#   INPUT       – path to the original counts matrix (sgb_table.tsv)
#                 (needed for RMSE/MAE metrics and NMF baseline)
#   LABEL       – metadata column for colouring ordination points
#                 (default: "disease")
#   STUDY_NAME  – short identifier used in figure titles (default: "Study")
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODELS_DIR="${1:?Usage: $0 <MODELS_DIR> <METADATA> <FIGURES_DIR> [INPUT] [LABEL] [STUDY_NAME]}"
METADATA="${2:?Missing METADATA path}"
FIGURES_DIR="${3:?Missing FIGURES_DIR path}"
INPUT="${4:-}"
LABEL="${5:-disease}"
STUDY_NAME="${6:-Study}"

# ── logging ──────────────────────────────────────────────────────────
LOG_DIR="${FIGURES_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/figures_$(date +'%Y%m%d_%H%M%S').log"

log_with_timestamp() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "Starting figure generation"
log_with_timestamp "  MODELS_DIR  = ${MODELS_DIR}"
log_with_timestamp "  METADATA    = ${METADATA}"
log_with_timestamp "  FIGURES_DIR = ${FIGURES_DIR}"
log_with_timestamp "  LABEL       = ${LABEL}"
log_with_timestamp "  STUDY_NAME  = ${STUDY_NAME}"

# ── conda / mamba environment ────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

mkdir -p "${FIGURES_DIR}"

CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-single-study-figures
  --results-dir "${MODELS_DIR}"
  --metadata "${METADATA}"
  --outdir "${FIGURES_DIR}"
  --label "${LABEL}"
  --study-name "${STUDY_NAME}"
)

if [[ -n "${INPUT}" && -f "${INPUT}" ]]; then
  CMD+=(--input "${INPUT}")
fi

log_with_timestamp "Running: ${CMD[*]}"
if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  log_with_timestamp "Figure generation completed successfully."
else
  log_with_timestamp "ERROR: figure generation failed (exit ${PIPESTATUS[0]})."
  exit 1
fi
