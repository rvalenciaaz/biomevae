#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# classify_model.sh – run disease classification for a single biomevae model
#
# Usage (called by classify_model.slurm via srun):
#   ./classify_model.sh <MODEL_OUTDIR> <METADATA> <LABEL>
#
# Arguments:
#   MODEL_OUTDIR – output directory from training (contains embed/ or test/)
#   METADATA     – path to sample metadata TSV (must contain LABEL column)
#   LABEL        – metadata column to classify on (e.g. "disease")
#
# The script looks for embeddings in <MODEL_OUTDIR>/embed/embeddings.tsv
# or <MODEL_OUTDIR>/test/embeddings.tsv and writes results to
# <MODEL_OUTDIR>/classify/.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_OUTDIR="${1:?Usage: $0 <MODEL_OUTDIR> <METADATA> <LABEL>}"
METADATA="${2:?Missing METADATA path}"
LABEL="${3:-disease}"

# ── evaluation seeds ─────────────────────────────────────────────────
# Classification is pooled across these seeds for reproducibility.
# Kept in sync with ``biomevae.classify.DEFAULT_EVAL_SEEDS``. Override via
# the ``EVAL_SEEDS`` environment variable.
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
read -r -a EVAL_SEEDS_ARRAY <<< "${EVAL_SEEDS}"

MODEL_NAME="$(basename "${MODEL_OUTDIR}")"

# ── logging ──────────────────────────────────────────────────────────
LOG_DIR="${MODEL_OUTDIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/classify_$(date +'%Y%m%d_%H%M%S').log"

log_with_timestamp() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "Starting classification for: ${MODEL_OUTDIR}"

# ── conda / mamba environment ────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

# ── find embeddings ─────────────────────────────────────────────────
EMBED_FILE=""
for candidate in "${MODEL_OUTDIR}/embed/embeddings.tsv" "${MODEL_OUTDIR}/test/embeddings.tsv" "${MODEL_OUTDIR}/embeddings.tsv"; do
  if [[ -f "${candidate}" ]]; then
    EMBED_FILE="${candidate}"
    break
  fi
done

if [[ -z "${EMBED_FILE}" ]]; then
  log_with_timestamp "SKIP ${MODEL_NAME}: no embeddings found."
  exit 0
fi

CLASSIFY_DIR="${MODEL_OUTDIR}/classify"

if [[ -f "${CLASSIFY_DIR}/classification_results.json" ]]; then
  log_with_timestamp "SKIP ${MODEL_NAME}: classification already complete."
  exit 0
fi

# ── run classification ──────────────────────────────────────────────
mkdir -p "${CLASSIFY_DIR}"
CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-classify
  --embeddings "${EMBED_FILE}"
  --metadata "${METADATA}"
  --label "${LABEL}"
  --outdir "${CLASSIFY_DIR}"
  --n-splits 5
  --n-repeats 10
  --seeds "${EVAL_SEEDS_ARRAY[@]}"
)

log_with_timestamp "Running: ${CMD[*]}"
if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  log_with_timestamp "Classification completed successfully for ${MODEL_NAME}."
else
  log_with_timestamp "WARNING: classification failed for ${MODEL_NAME} (exit ${PIPESTATUS[0]})."
  exit 1
fi
