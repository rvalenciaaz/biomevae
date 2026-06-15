#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# single_study_pipeline.sh – HPC pipeline for a single microbiome study
#
# Runs the full analysis on an HPC cluster (no internet required).
# Data must be pre-extracted on a machine with internet access using the
# extract-microbiome-data package:
#   https://github.com/rvazdev-ex/extract-microbiome-data
#
# The expected layout of --data-dir is:
#   <data-dir>/sgb_table.tsv        (abundance table)
#   <data-dir>/phyla.tsv            (taxonomy mapping)
#   <data-dir>/sample_metadata.tsv  (sample metadata)
#
# Pipeline steps:
#   1. TRAIN       – Train all VAE model variants (parallel SLURM jobs)
#   2. POSTPROCESS – Extract embeddings, evaluate reconstruction, SHAP
#   3. CLASSIFY    – Classification on the chosen metadata label from
#                    the VAE embeddings (plus an XGBoost baseline from
#                    the raw SGB table)
#   4. FIGURES     – Generate publication-quality figures and results tables
#   5. AGGREGATE   – Cross-model benchmarking (allcomp with NMF baseline,
#                    benchmark figures, violin plots, pairwise significance,
#                    hierarchy metrics, scatter plots, enterosignatures,
#                    slides, SHAP comparison)
#
# Usage:
#   ./single_study_pipeline.sh \
#       --study-name LiJ_2017 \
#       --outdir /scratch/lij_2017 \
#       --data-dir /path/to/lij_2017_data
#
# Required:
#   --outdir      Root output directory for models, figures, etc.
#   --data-dir    Directory containing pre-extracted data files:
#                   sgb_table.tsv, phyla.tsv, sample_metadata.tsv
#                 (produced by the extract-microbiome-data package)
#
# Optional:
#   --study-name       Short identifier used for SLURM job names and
#                      figure titles (default: "study")
#   --label            Metadata column to classify on (default: "disease")
#   --extra-args       Extra arguments for model training
#                      (default: "--epochs 100 --optuna --optuna-trials 100")
#   --skip-train       Skip model training (re-use existing models)
#   --skip-postprocess Skip postprocessing (re-use existing embeddings/test results)
#   --skip-classify    Skip classification (re-use existing classification results)
#   --skip-aggregate   Skip aggregation/benchmarking (re-use existing aggregate results)
#   --dry-run          Print commands without executing
#   -h, --help         Show this help message
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────
OUTDIR=""
DATA_DIR=""
STUDY_NAME="study"
LABEL="disease"
EXTRA_ARGS="--epochs 100 --optuna --optuna-trials 100"
SKIP_TRAIN=false
SKIP_POSTPROCESS=false
SKIP_CLASSIFY=false
SKIP_AGGREGATE=false
CLASSIFY_JOBIDS_CSV=""
DRY_RUN=false

usage() {
  head -n 52 "$0" | tail -n +2 | sed 's/^# \?//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)           OUTDIR="$2";           shift 2 ;;
    --data-dir)         DATA_DIR="$2";         shift 2 ;;
    --study-name)       STUDY_NAME="$2";       shift 2 ;;
    --label)            LABEL="$2";            shift 2 ;;
    --extra-args)       EXTRA_ARGS="$2";       shift 2 ;;
    --skip-train)       SKIP_TRAIN=true;       shift   ;;
    --skip-postprocess) SKIP_POSTPROCESS=true; shift   ;;
    --skip-classify)    SKIP_CLASSIFY=true;    shift   ;;
    --skip-aggregate)   SKIP_AGGREGATE=true;   shift   ;;
    --dry-run)          DRY_RUN=true;          shift   ;;
    -h|--help)      usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ -z "$OUTDIR"   ]] && { echo "Error: --outdir is required."   >&2; usage 1; }
[[ -z "$DATA_DIR" ]] && { echo "Error: --data-dir is required." >&2; usage 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sanitise the study name for use in SLURM job names / file paths.
STUDY_SLUG="$(echo "${STUDY_NAME}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\+/-/g; s/^-\+//; s/-\+$//')"
[[ -z "${STUDY_SLUG}" ]] && STUDY_SLUG="study"

# ── Validate pre-extracted data ─────────────────────────────────────────
INPUT="${DATA_DIR}/sgb_table.tsv"
TAXONOMY="${DATA_DIR}/phyla.tsv"
METADATA="${DATA_DIR}/sample_metadata.tsv"

for f in "${INPUT}" "${TAXONOMY}" "${METADATA}"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: Required data file not found: $f" >&2
    echo "" >&2
    echo "Data must be pre-extracted using the extract-microbiome-data package:" >&2
    echo "  https://github.com/rvazdev-ex/extract-microbiome-data" >&2
    echo "" >&2
    echo "Example (curatedMetagenomicData single study):" >&2
    echo "  python -m curatedmetagenomicdata.extract \\" >&2
    echo "      --study ${STUDY_NAME} -o ${DATA_DIR}" >&2
    exit 1
  fi
done

# ── Conda / mamba environment ───────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

run_cmd() {
  if $DRY_RUN; then
    echo "[DRY-RUN] ${MAMBA_EXEC} run -n ${CONDA_ENV} $*"
  else
    "${MAMBA_EXEC}" run -n "${CONDA_ENV}" "$@"
  fi
}

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# ── Directory structure ─────────────────────────────────────────────────
MODELS_DIR="${OUTDIR}/models"
FIGURES_DIR="${OUTDIR}/figures"
mkdir -p "${MODELS_DIR}" "${FIGURES_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo " Single-study Analysis Pipeline (HPC)"
echo "═══════════════════════════════════════════════════════════════"
echo " STUDY_NAME   : ${STUDY_NAME}"
echo " DATA_DIR     : ${DATA_DIR}"
echo " OUTDIR       : ${OUTDIR}"
echo " MODELS_DIR   : ${MODELS_DIR}"
echo " FIGURES_DIR  : ${FIGURES_DIR}"
echo " LABEL        : ${LABEL}"
echo " EXTRA_ARGS   : ${EXTRA_ARGS:-<none>}"
echo " DRY RUN      : ${DRY_RUN}"
echo "───────────────────────────────────────────────────────────────"

# ═════════════════════════════════════════════════════════════════════════
# STEP 1: MODEL TRAINING (parallel SLURM jobs)
# ═════════════════════════════════════════════════════════════════════════

if ! $SKIP_TRAIN; then
  log "STEP 1: Submitting model training jobs..."

  SUBMIT_ARGS=(
    "${SCRIPT_DIR}/submit_all.sh"
    --input "${INPUT}"
    --taxonomy "${TAXONOMY}"
    --outdir "${MODELS_DIR}"
  )
  if [[ -n "${EXTRA_ARGS}" ]]; then
    SUBMIT_ARGS+=(--extra-args "${EXTRA_ARGS}")
  fi
  if $DRY_RUN; then
    SUBMIT_ARGS+=(--dry-run)
  fi

  "${SUBMIT_ARGS[@]}"

  if ! $DRY_RUN; then
    log "  Training jobs submitted. Monitor with: squeue -u \$USER"
    log ""
    log "  ╔══════════════════════════════════════════════════════════╗"
    log "  ║  Wait for all training jobs to complete, then re-run    ║"
    log "  ║  this script with --skip-train to continue with         ║"
    log "  ║  postprocessing, classification, and figure generation. ║"
    log "  ║                                                         ║"
    log "  ║  Alternatively, submit a dependent job:                 ║"
    log "  ║    sbatch --dependency=afterok:<JOB_IDS>                ║"
    log "  ║      single_study_pipeline.slurm                        ║"
    log "  ╚══════════════════════════════════════════════════════════╝"
    log ""

    # If running interactively (not in SLURM), suggest next steps and exit
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      log "Exiting. Re-run with --skip-train after training completes."
      exit 0
    fi
  fi
else
  log "STEP 1: Skipped (--skip-train). Using existing models in ${MODELS_DIR}"
fi

# ═════════════════════════════════════════════════════════════════════════
# STEP 2: POSTPROCESSING (embeddings, test, interpret) – parallel SLURM jobs
# ═════════════════════════════════════════════════════════════════════════

if ! $SKIP_POSTPROCESS; then
  log "STEP 2: Submitting postprocessing SLURM jobs..."

  PP_ARGS=(
    "${SCRIPT_DIR}/submit_all_postprocess.sh"
    --input "${INPUT}"
    --taxonomy "${TAXONOMY}"
    --outdir "${MODELS_DIR}"
  )
  if $DRY_RUN; then
    PP_ARGS+=(--dry-run)
  fi

  "${PP_ARGS[@]}"

  if ! $DRY_RUN; then
    log "  Postprocessing jobs submitted. Monitor with: squeue -u \$USER"
    log ""
    log "  ╔══════════════════════════════════════════════════════════╗"
    log "  ║  Wait for all postprocessing jobs to complete, then     ║"
    log "  ║  re-run this script with --skip-train --skip-postprocess║"
    log "  ║  to continue with classification and figure generation. ║"
    log "  ╚══════════════════════════════════════════════════════════╝"
    log ""

    # If running interactively (not in SLURM), suggest next steps and exit
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      log "Exiting. Re-run with --skip-train --skip-postprocess after postprocessing completes."
      exit 0
    fi
  fi
else
  log "STEP 2: Skipped (--skip-postprocess). Using existing results in ${MODELS_DIR}"
fi

# ═════════════════════════════════════════════════════════════════════════
# STEP 3: CLASSIFICATION (on the chosen metadata label) – parallel SLURM jobs
# ═════════════════════════════════════════════════════════════════════════

if ! $SKIP_CLASSIFY; then
  log "STEP 3: Submitting classification SLURM jobs (label=${LABEL})..."

  CLF_ARGS=(
    "${SCRIPT_DIR}/submit_all_classify.sh"
    --metadata "${METADATA}"
    --outdir "${MODELS_DIR}"
    --input "${INPUT}"
    --label "${LABEL}"
  )
  if $DRY_RUN; then
    CLF_ARGS+=(--dry-run)
  fi

  # Capture output to extract job IDs for figure dependency chaining
  CLF_OUTPUT=$("${CLF_ARGS[@]}" | tee /dev/stderr)
  CLASSIFY_JOBIDS_CSV="$(echo "${CLF_OUTPUT}" | grep -oP 'CLASSIFY_JOBIDS=\K.*' || true)"

  if ! $DRY_RUN; then
    log "  Classification jobs submitted. Monitor with: squeue -u \$USER"
  fi
else
  log "STEP 3: Skipped (--skip-classify). Using existing results in ${MODELS_DIR}"
fi

# ═════════════════════════════════════════════════════════════════════════
# STEP 4: FIGURE GENERATION – SLURM job (depends on classification)
# ═════════════════════════════════════════════════════════════════════════

log "STEP 4: Submitting figure generation SLURM job..."

FIG_SLURM="${SCRIPT_DIR}/generate_figures.slurm"
FIG_SCRIPT="${SCRIPT_DIR}/generate_figures.sh"

FIG_SBATCH_CMD=(
  sbatch
  --job-name="${STUDY_SLUG}-figures"
  --output="${FIGURES_DIR}/slurm_fig_%j.out"
  --error="${FIGURES_DIR}/slurm_fig_%j.err"
  --export="ALL,MODELS_DIR=${MODELS_DIR},METADATA=${METADATA},FIGURES_DIR=${FIGURES_DIR},FIG_SCRIPT=${FIG_SCRIPT},INPUT=${INPUT},LABEL=${LABEL},STUDY_NAME=${STUDY_NAME}"
)

# Chain figure generation after classification jobs if they were submitted
if ! $SKIP_CLASSIFY && ! $DRY_RUN && [[ -n "${CLASSIFY_JOBIDS_CSV}" ]]; then
  FIG_SBATCH_CMD+=(--dependency="afterok:${CLASSIFY_JOBIDS_CSV//,/:}")
fi

FIG_SBATCH_CMD+=("${FIG_SLURM}")

if $DRY_RUN; then
  log "  [DRY-RUN] ${FIG_SBATCH_CMD[*]}"
else
  mkdir -p "${FIGURES_DIR}"
  log "  Submitting figure generation job..."
  FIG_OUTPUT=$("${FIG_SBATCH_CMD[@]}" 2>&1)
  log "  ${FIG_OUTPUT}"
fi

# ═════════════════════════════════════════════════════════════════════════
# STEP 5: AGGREGATION / BENCHMARKING – allcomp with NMF baseline,
#         benchmark figures, violin plots, pairwise tables, etc.
# ═════════════════════════════════════════════════════════════════════════

if ! $SKIP_AGGREGATE; then
  log "STEP 5: Submitting aggregation / benchmarking SLURM job..."

  AGG_SLURM="${SCRIPT_DIR}/aggregate_results.slurm"
  AGG_SCRIPT="${SCRIPT_DIR}/aggregate_results.sh"

  if [[ ! -f "${AGG_SLURM}" ]]; then
    log "  WARNING: ${AGG_SLURM} not found; skipping aggregation."
  elif [[ ! -f "${AGG_SCRIPT}" ]]; then
    log "  WARNING: ${AGG_SCRIPT} not found; skipping aggregation."
  else
    AGGREGATE_DIR="${MODELS_DIR}/aggregate"

    AGG_SBATCH_CMD=(
      sbatch
      --job-name="${STUDY_SLUG}-aggregate"
      --output="${AGGREGATE_DIR}/slurm_agg_%j.out"
      --error="${AGGREGATE_DIR}/slurm_agg_%j.err"
      --export="ALL,OUTDIR=${MODELS_DIR},INPUT=${INPUT},TAXONOMY=${TAXONOMY},AGG_SCRIPT=${AGG_SCRIPT}"
    )

    # Chain aggregation after classification jobs (ensures postprocessing is done)
    if ! $SKIP_CLASSIFY && ! $DRY_RUN && [[ -n "${CLASSIFY_JOBIDS_CSV}" ]]; then
      AGG_SBATCH_CMD+=(--dependency="afterany:${CLASSIFY_JOBIDS_CSV//,/:}")
    fi

    AGG_SBATCH_CMD+=("${AGG_SLURM}")

    if $DRY_RUN; then
      log "  [DRY-RUN] ${AGG_SBATCH_CMD[*]}"
    else
      mkdir -p "${AGGREGATE_DIR}"
      log "  Submitting aggregation job..."
      AGG_OUTPUT=$("${AGG_SBATCH_CMD[@]}" 2>&1)
      log "  ${AGG_OUTPUT}"
    fi
  fi
else
  log "STEP 5: Skipped (--skip-aggregate). Using existing results in ${MODELS_DIR}/aggregate"
fi

# ═════════════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════════"
log " All jobs submitted!"
log "═══════════════════════════════════════════════════════════════"
log ""
log " Monitor progress with: squeue -u \$USER"
log ""
log " Data:     ${DATA_DIR}/"
log " Models:   ${MODELS_DIR}/"
log " Figures:  ${FIGURES_DIR}/"
log ""
log " Key outputs (after all jobs complete):"
log "   ${FIGURES_DIR}/fig1_latent_ordination.pdf"
log "   ${FIGURES_DIR}/fig2_classification_performance.pdf"
log "   ${FIGURES_DIR}/fig3_confusion_matrices.pdf"
log "   ${FIGURES_DIR}/fig4_reconstruction_quality.pdf"
log "   ${FIGURES_DIR}/fig5_training_curves.pdf"
log "   ${FIGURES_DIR}/results_summary.tsv"
log "   ${FIGURES_DIR}/results_summary.tex"
log ""
log " XGBoost baseline outputs:"
log "   ${MODELS_DIR}/xgboost-baseline/classify/xgboost_baseline_classification_results.json"
log ""
log " Aggregate benchmarking outputs:"
log "   ${MODELS_DIR}/aggregate/all_methods_vs_nmf.json"
log "   ${MODELS_DIR}/aggregate/test_summary.tsv"
log "   ${MODELS_DIR}/aggregate/figures/benchmark_metrics.pdf"
log "   ${MODELS_DIR}/aggregate/figures/benchmark_violin.pdf"
log "   ${MODELS_DIR}/aggregate/figures/benchmark_ordinations.pdf"
log ""
