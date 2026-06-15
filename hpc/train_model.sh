#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# train_model.sh – run a single biomevae model training inside a SLURM job
#
# Usage (called by train_model.slurm via srun):
#   ./train_model.sh <MODEL_CMD> <INPUT> <OUTDIR> [TAXONOMY] [EXTRA_ARGS...]
#
# Arguments:
#   MODEL_CMD   – CLI entry point (e.g. biomevae-train, biomevae-train-tax, …)
#   INPUT       – path to the abundance table (TSV/CSV)
#   OUTDIR      – output directory for this model run
#   TAXONOMY    – path to taxonomy file (required for taxonomy-aware models,
#                 pass "none" to skip)
#   EXTRA_ARGS  – any additional CLI flags forwarded verbatim
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_CMD="${1:?Usage: $0 <MODEL_CMD> <INPUT> <OUTDIR> [TAXONOMY] [EXTRA_ARGS...]}"
INPUT="${2:?Missing INPUT path}"
OUTDIR="${3:?Missing OUTDIR path}"
TAXONOMY="${4:-none}"
shift 4 || true
EXTRA_ARGS_ENV="${EXTRA_ARGS:-}"
EXTRA_ARGS_B64_ENV="${EXTRA_ARGS_B64:-}"
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]] && [[ -n "${EXTRA_ARGS_B64_ENV}" ]]; then
  EXTRA_ARGS_ENV="$(printf '%s' "${EXTRA_ARGS_B64_ENV}" | base64 --decode)"
fi
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]] && [[ -n "${EXTRA_ARGS_ENV}" ]]; then
  while IFS= read -r -d '' arg; do
    [[ -z "${arg}" ]] && continue
    EXTRA_ARGS+=("$arg")
  done < <(
    EXTRA_ARGS_STRING="${EXTRA_ARGS_ENV}" python3 - <<'PY'
import os, shlex, sys
parts = shlex.split(os.environ.get("EXTRA_ARGS_STRING", ""))
if parts:
    sys.stdout.buffer.write(b"\0".join(p.encode("utf-8") for p in parts))
    sys.stdout.buffer.write(b"\0")
PY
  )
fi

# ── logging ──────────────────────────────────────────────────────────
LOG_DIR="${OUTDIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/log_$(date +'%Y%m%d_%H%M%S').log"

log_with_timestamp() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "Starting training: ${MODEL_CMD}"
log_with_timestamp "  INPUT    = ${INPUT}"
log_with_timestamp "  OUTDIR   = ${OUTDIR}"
log_with_timestamp "  TAXONOMY = ${TAXONOMY}"
log_with_timestamp "  EXTRA    = ${EXTRA_ARGS[*]:-<none>}"

# ── conda / mamba environment ────────────────────────────────────────
# Adjust these paths to match your HPC environment.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

# ── build command ────────────────────────────────────────────────────
CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" "${MODEL_CMD}"
     --input "${INPUT}"
     --outdir "${OUTDIR}")

if [[ "${TAXONOMY}" != "none" ]]; then
  CMD+=(--taxonomy "${TAXONOMY}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

# ── run ──────────────────────────────────────────────────────────────
log_with_timestamp "Running: ${CMD[*]}"
set +eo pipefail
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -eo pipefail

if [[ $STATUS -ne 0 ]]; then
  log_with_timestamp "Error: ${MODEL_CMD} exited with status ${STATUS}."
  exit "$STATUS"
fi

log_with_timestamp "Training completed successfully: ${MODEL_CMD}"
exit 0
