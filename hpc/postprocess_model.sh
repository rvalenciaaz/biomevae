#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# postprocess_model.sh – run post-training steps for a single biomevae model
#
# Usage (called by postprocess_model.slurm via srun):
#   ./postprocess_model.sh <MODEL_OUTDIR> <INPUT> [TAXONOMY] [EXTRA_ARGS...]
#
# Arguments:
#   MODEL_OUTDIR – output directory from training (contains model.pt & config.json)
#   INPUT        – path to the abundance table (TSV/CSV)
#   TAXONOMY     – path to taxonomy file (pass "none" to skip taxonomy-dependent steps)
#   EXTRA_ARGS   – any additional CLI flags forwarded to individual commands
#
# Steps executed:
#   1. biomevae-test  --export  (evaluation metrics + embeddings + reconstruction)
#   2. biomevae-embed           (standalone embedding extraction)
#   3. biomevae-interpret       (SHAP interpretation, skipped for unsupported model types)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_OUTDIR="${1:?Usage: $0 <MODEL_OUTDIR> <INPUT> [TAXONOMY] [EXTRA_ARGS...]}"
INPUT="${2:?Missing INPUT path}"
TAXONOMY="${3:-none}"
shift 3 || true
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
    python3 - <<'PY'
import os, shlex, sys
parts = shlex.split(os.environ.get("EXTRA_ARGS", ""))
if parts:
    sys.stdout.buffer.write(b"\0".join(p.encode("utf-8") for p in parts))
    sys.stdout.buffer.write(b"\0")
PY
  )
fi

# ── validate training artifacts ─────────────────────────────────────
if [[ ! -f "${MODEL_OUTDIR}/model.pt" ]]; then
  echo "Error: model.pt not found in ${MODEL_OUTDIR}. Training may not have completed." >&2
  exit 1
fi

if [[ ! -f "${MODEL_OUTDIR}/config.json" ]]; then
  echo "Error: config.json not found in ${MODEL_OUTDIR}." >&2
  exit 1
fi

# ── logging ──────────────────────────────────────────────────────────
LOG_DIR="${MODEL_OUTDIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/postprocess_$(date +'%Y%m%d_%H%M%S').log"

log_with_timestamp() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "Starting post-processing for: ${MODEL_OUTDIR}"
log_with_timestamp "  INPUT    = ${INPUT}"
log_with_timestamp "  TAXONOMY = ${TAXONOMY}"
log_with_timestamp "  EXTRA    = ${EXTRA_ARGS[*]:-<none>}"

# ── conda / mamba environment ────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

# ── detect model type from config.json ──────────────────────────────
MODEL_TYPE=$("${MAMBA_EXEC}" run -n "${CONDA_ENV}" python -c "
import json, sys
with open('${MODEL_OUTDIR}/config.json') as f:
    cfg = json.load(f)
print(cfg.get('model_type', 'euclid'))
" 2>/dev/null || echo "unknown")

log_with_timestamp "Detected model_type: ${MODEL_TYPE}"

# ── helper: build taxonomy args ─────────────────────────────────────
tax_args=()
if [[ "${TAXONOMY}" != "none" ]]; then
  tax_args=(--taxonomy "${TAXONOMY}")
fi

# ── output directories ──────────────────────────────────────────────
TEST_OUTDIR="${MODEL_OUTDIR}/test"
EMBED_OUTDIR="${MODEL_OUTDIR}/embed"
INTERPRET_OUTDIR="${MODEL_OUTDIR}/interpret"

# ── model types that biomevae-interpret does NOT support ──────────────
# The interpret CLI handles: euclid, hyperbolic, graph_tax,
# treeprior, phylo_fusion, tree-dtm-vae, philrvae.  All others are
# skipped gracefully.
INTERPRET_UNSUPPORTED="hgvae_zi flowxformer"

# ── model types that biomevae-test does NOT support ───────────────────
# flowxformer falls through to the VAE else-branch in vae_test.py and
# would fail.  Skip it gracefully.
TEST_UNSUPPORTED="flowxformer"

# ════════════════════════════════════════════════════════════════════
# Step 1: biomevae-test  (evaluation + export)
# ════════════════════════════════════════════════════════════════════
run_test() {
  if echo "${TEST_UNSUPPORTED}" | grep -qw "${MODEL_TYPE}"; then
    log_with_timestamp "SKIP biomevae-test: model_type '${MODEL_TYPE}' is not supported."
    return 0
  fi

  mkdir -p "${TEST_OUTDIR}"
  local CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-test
    --input "${INPUT}"
    --model-dir "${MODEL_OUTDIR}"
    --outdir "${TEST_OUTDIR}"
    --export
    "${tax_args[@]}"
  )
  log_with_timestamp "Running biomevae-test: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "biomevae-test completed successfully."
  else
    log_with_timestamp "WARNING: biomevae-test failed (exit ${PIPESTATUS[0]}). Continuing."
  fi
}

# ════════════════════════════════════════════════════════════════════
# Step 2: biomevae-embed (standalone embedding extraction)
# ════════════════════════════════════════════════════════════════════
run_embed() {
  mkdir -p "${EMBED_OUTDIR}"
  local CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-embed
    --input "${INPUT}"
    --model-dir "${MODEL_OUTDIR}"
    --outdir "${EMBED_OUTDIR}"
    --export-recon
    "${tax_args[@]}"
  )
  log_with_timestamp "Running biomevae-embed: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "biomevae-embed completed successfully."
  else
    log_with_timestamp "WARNING: biomevae-embed failed (exit ${PIPESTATUS[0]}). Continuing."
  fi
}

# ════════════════════════════════════════════════════════════════════
# Step 3: biomevae-interpret (SHAP-based interpretation)
# ════════════════════════════════════════════════════════════════════
run_interpret() {
  if echo "${INTERPRET_UNSUPPORTED}" | grep -qw "${MODEL_TYPE}"; then
    log_with_timestamp "SKIP biomevae-interpret: model_type '${MODEL_TYPE}' is not supported."
    return 0
  fi

  mkdir -p "${INTERPRET_OUTDIR}"
  local CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-interpret
    --input "${INPUT}"
    --model-dir "${MODEL_OUTDIR}"
    --outdir "${INTERPRET_OUTDIR}"
    "${tax_args[@]}"
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  log_with_timestamp "Running biomevae-interpret: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "biomevae-interpret completed successfully."
  else
    log_with_timestamp "WARNING: biomevae-interpret failed (exit ${PIPESTATUS[0]}). Continuing."
  fi
}

# ════════════════════════════════════════════════════════════════════
# Step 4: biomevae-interpret at genus level (taxonomy aggregation)
# ════════════════════════════════════════════════════════════════════
run_interpret_genus() {
  if echo "${INTERPRET_UNSUPPORTED}" | grep -qw "${MODEL_TYPE}"; then
    log_with_timestamp "SKIP biomevae-interpret (genus): model_type '${MODEL_TYPE}' is not supported."
    return 0
  fi

  if [[ "${TAXONOMY}" == "none" ]]; then
    log_with_timestamp "SKIP biomevae-interpret (genus): no taxonomy provided."
    return 0
  fi

  local INTERPRET_GENUS_OUTDIR="${MODEL_OUTDIR}/interpret_genus"
  mkdir -p "${INTERPRET_GENUS_OUTDIR}"
  local CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-interpret
    --input "${INPUT}"
    --model-dir "${MODEL_OUTDIR}"
    --outdir "${INTERPRET_GENUS_OUTDIR}"
    --taxonomy-level genus
    "${tax_args[@]}"
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  log_with_timestamp "Running biomevae-interpret (genus): ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "biomevae-interpret (genus) completed successfully."
  else
    log_with_timestamp "WARNING: biomevae-interpret (genus) failed (exit ${PIPESTATUS[0]}). Continuing."
  fi
}

# ── execute steps ────────────────────────────────────────────────────
run_test
run_embed
run_interpret
run_interpret_genus

log_with_timestamp "Post-processing completed for: ${MODEL_OUTDIR}"
exit 0
