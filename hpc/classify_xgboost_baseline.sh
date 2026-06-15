#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# classify_xgboost_baseline.sh – run XGBoost directly on the SGB table
#
# Usage (called by classify_xgboost_baseline.slurm via srun):
#   ./classify_xgboost_baseline.sh <INPUT> <METADATA> <LABEL> <OUTDIR>
#
# Arguments:
#   INPUT    – path to sgb_table.tsv
#   METADATA – path to sample metadata TSV (must contain LABEL column)
#   LABEL    – metadata column to classify on (e.g. "disease")
#   OUTDIR   – root output directory (results go to <OUTDIR>/xgboost-baseline/classify/)
#
# This baseline classifies directly from abundances without any
# dimensionality reduction, serving as a reference for VAE-based
# classification performance.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

INPUT="${1:?Usage: $0 <INPUT> <METADATA> <LABEL> <OUTDIR>}"
METADATA="${2:?Missing METADATA path}"
LABEL="${3:-disease}"
OUTDIR="${4:?Missing OUTDIR path}"

# ── evaluation seeds ─────────────────────────────────────────────────
# Classification is pooled across these seeds for reproducibility.
# Kept in sync with ``biomevae.classify.DEFAULT_EVAL_SEEDS``. Override via
# the ``EVAL_SEEDS`` environment variable.
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
read -r -a EVAL_SEEDS_ARRAY <<< "${EVAL_SEEDS}"

BASELINE_DIR="${OUTDIR}/xgboost-baseline"
CLASSIFY_DIR="${BASELINE_DIR}/classify"

# ── logging ──────────────────────────────────────────────────────────
LOG_DIR="${BASELINE_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/classify_$(date +'%Y%m%d_%H%M%S').log"

log_with_timestamp() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "Starting XGBoost baseline classification"
log_with_timestamp "  INPUT:    ${INPUT}"
log_with_timestamp "  METADATA: ${METADATA}"
log_with_timestamp "  LABEL:    ${LABEL}"

# ── conda / mamba environment ────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

if [[ -f "${CLASSIFY_DIR}/xgboost_baseline_classification_results.json" ]]; then
  log_with_timestamp "SKIP xgboost-baseline: classification already complete."
  exit 0
fi

# ── run classification ──────────────────────────────────────────────
mkdir -p "${CLASSIFY_DIR}"
CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-classify-baseline
  --input "${INPUT}"
  --metadata "${METADATA}"
  --label "${LABEL}"
  --outdir "${CLASSIFY_DIR}"
  --log1p
  --n-splits 5
  --n-repeats 10
  --seeds "${EVAL_SEEDS_ARRAY[@]}"
)

log_with_timestamp "Running: ${CMD[*]}"
if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  log_with_timestamp "XGBoost baseline classification completed successfully."
else
  log_with_timestamp "WARNING: XGBoost baseline classification failed (exit ${PIPESTATUS[0]})."
  exit 1
fi
