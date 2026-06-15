#!/usr/bin/env bash
set -euo pipefail

METADATA=""
OUTDIR=""
INPUT=""
LABEL="disease"
DRY_RUN=false
PP_JOBIDS=""

usage() {
  cat <<EOF
Usage: $0 --metadata <FILE> --outdir <DIR> [OPTIONS]

Submit classification jobs for all trained biomevae models.

Required:
  --metadata   Path to sample metadata TSV (must contain the label column)
  --outdir     Root output directory (must match the --outdir used for training)

Optional:
  --input        Path to sgb_table.tsv (required for XGBoost baseline job)
  --label        Metadata column to classify on (default: "disease")
  --pp-jobids    Comma-separated SLURM job IDs from postprocessing submission.
                 Classification jobs will depend on the corresponding
                 postprocessing job via --dependency=afterok:<jobid>.
                 Order must match the model list (beta-vae, vanilla-vae,
                 hyp-vae, tax-vae, hyp-tax-vae, graph-vae, treeprior-vae,
                 fuse-vae, tree-dtm-vae, philrvae, hyp-philrvae,
                 hyp-philr-zinb).
  --dry-run      Print sbatch commands without submitting
  -h, --help     Show this help message
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metadata)   METADATA="$2";   shift 2 ;;
    --outdir)     OUTDIR="$2";     shift 2 ;;
    --input)      INPUT="$2";      shift 2 ;;
    --label)      LABEL="$2";      shift 2 ;;
    --pp-jobids)  PP_JOBIDS="$2";  shift 2 ;;
    --dry-run)    DRY_RUN=true;    shift   ;;
    -h|--help)    usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ -z "$METADATA" ]] && { echo "Error: --metadata is required." >&2; usage 1; }
[[ -z "$OUTDIR"   ]] && { echo "Error: --outdir is required."   >&2; usage 1; }
[[ ! -f "$METADATA" ]] && { echo "Error: --metadata file not found: $METADATA" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/classify_model.slurm"
CLF_SCRIPT="${SCRIPT_DIR}/classify_model.sh"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Error: cannot find ${SLURM_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "$CLF_SCRIPT" ]]; then
  echo "Error: cannot find ${CLF_SCRIPT}" >&2
  exit 1
fi

# Same model list as submit_all.sh
MODEL_NAMES=(
  beta-vae
  vanilla-vae
  hyp-vae
  tax-vae
  hyp-tax-vae
  graph-vae
  treeprior-vae
  fuse-vae
  tree-dtm-vae
  philrvae
  hyp-philrvae
  hyp-philr-zinb
)

# Parse --pp-jobids into an array (empty if not provided)
IFS=',' read -ra JOBID_ARRAY <<< "${PP_JOBIDS}"
if [[ -n "${PP_JOBIDS}" ]] && [[ ${#JOBID_ARRAY[@]} -ne ${#MODEL_NAMES[@]} ]]; then
  echo "Error: --pp-jobids must contain exactly ${#MODEL_NAMES[@]} comma-separated IDs (one per model)." >&2
  exit 1
fi

echo "============================================================"
echo " biomevae – parallel HPC classification submission"
echo "============================================================"
echo " METADATA : ${METADATA}"
echo " OUTDIR   : ${OUTDIR}"
echo " LABEL    : ${LABEL}"
echo " JOBIDS   : ${PP_JOBIDS:-<none>}"
echo " DRY RUN  : ${DRY_RUN}"
echo "------------------------------------------------------------"

SUBMITTED=0
SKIPPED=0
CLASSIFY_JOBIDS=()

for i in "${!MODEL_NAMES[@]}"; do
  JOB_NAME="${MODEL_NAMES[$i]}"
  MODEL_OUTDIR="${OUTDIR}/${JOB_NAME}"

  # If no dependency was given, check that embeddings exist
  if [[ -z "${PP_JOBIDS}" ]]; then
    FOUND_EMBED=false
    for candidate in "${MODEL_OUTDIR}/embed/embeddings.tsv" "${MODEL_OUTDIR}/test/embeddings.tsv" "${MODEL_OUTDIR}/embeddings.tsv"; do
      if [[ -f "${candidate}" ]]; then
        FOUND_EMBED=true
        break
      fi
    done
    if ! $FOUND_EMBED; then
      echo "SKIP ${JOB_NAME}: no embeddings found in ${MODEL_OUTDIR}"
      SKIPPED=$((SKIPPED + 1))
      CLASSIFY_JOBIDS+=("")
      continue
    fi
  fi

  # Build sbatch command
  SBATCH_CMD=(
    sbatch
    --job-name="clf-${JOB_NAME}"
    --output="${MODEL_OUTDIR}/slurm_clf_%j.out"
    --error="${MODEL_OUTDIR}/slurm_clf_%j.err"
    --export="ALL,MODEL_OUTDIR=${MODEL_OUTDIR},METADATA=${METADATA},LABEL=${LABEL},CLF_SCRIPT=${CLF_SCRIPT}"
  )

  # Add SLURM dependency if a postprocessing job ID was provided for this model
  if [[ ${#JOBID_ARRAY[@]} -gt 0 ]] && [[ -n "${JOBID_ARRAY[$i]:-}" ]]; then
    SBATCH_CMD+=(--dependency="afterok:${JOBID_ARRAY[$i]}")
  fi

  SBATCH_CMD+=("${SLURM_SCRIPT}")

  if $DRY_RUN; then
    echo "[DRY-RUN] ${SBATCH_CMD[*]}"
    CLASSIFY_JOBIDS+=("DRY")
  else
    mkdir -p "${MODEL_OUTDIR}"
    echo -n "Submitting clf-${JOB_NAME}… "
    JOB_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1)
    echo "${JOB_OUTPUT}"
    # Extract job ID from "Submitted batch job 12345"
    JOB_ID="$(echo "${JOB_OUTPUT}" | grep -oP '\d+$' || echo "")"
    CLASSIFY_JOBIDS+=("${JOB_ID}")
  fi

  SUBMITTED=$((SUBMITTED + 1))
done

echo "------------------------------------------------------------"
echo " ${SUBMITTED} model classification jobs submitted, ${SKIPPED} skipped."

# ── XGBoost baseline (direct from SGB table) ────────────────────────
BASELINE_SLURM="${SCRIPT_DIR}/classify_xgboost_baseline.slurm"
CLF_BASELINE_SCRIPT="${SCRIPT_DIR}/classify_xgboost_baseline.sh"

if [[ -n "${INPUT}" && -f "${INPUT}" ]]; then
  if [[ -f "${BASELINE_SLURM}" && -f "${CLF_BASELINE_SCRIPT}" ]]; then
    BASELINE_OUTDIR="${OUTDIR}/xgboost-baseline"

    SBATCH_CMD=(
      sbatch
      --job-name="clf-xgboost-baseline"
      --output="${BASELINE_OUTDIR}/slurm_clf_%j.out"
      --error="${BASELINE_OUTDIR}/slurm_clf_%j.err"
      --export="ALL,INPUT=${INPUT},METADATA=${METADATA},LABEL=${LABEL},OUTDIR=${OUTDIR},CLF_BASELINE_SCRIPT=${CLF_BASELINE_SCRIPT}"
    )
    SBATCH_CMD+=("${BASELINE_SLURM}")

    if $DRY_RUN; then
      echo "[DRY-RUN] ${SBATCH_CMD[*]}"
      CLASSIFY_JOBIDS+=("DRY")
    else
      mkdir -p "${BASELINE_OUTDIR}"
      echo -n "Submitting clf-xgboost-baseline… "
      JOB_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1)
      echo "${JOB_OUTPUT}"
      JOB_ID="$(echo "${JOB_OUTPUT}" | grep -oP '\d+$' || echo "")"
      CLASSIFY_JOBIDS+=("${JOB_ID}")
    fi
    SUBMITTED=$((SUBMITTED + 1))
    echo " (includes XGBoost baseline job)"
  else
    echo " WARNING: XGBoost baseline SLURM/script not found; skipping baseline."
  fi
else
  echo " NOTE: --input not provided; skipping XGBoost baseline classification."
fi

echo "------------------------------------------------------------"
echo " ${SUBMITTED} total jobs submitted (model + baseline), ${SKIPPED} skipped."

# Print comma-separated job IDs for downstream dependency chaining
VALID_IDS=()
for jid in "${CLASSIFY_JOBIDS[@]}"; do
  [[ -n "$jid" ]] && [[ "$jid" != "DRY" ]] && VALID_IDS+=("$jid")
done
if [[ ${#VALID_IDS[@]} -gt 0 ]]; then
  JOINED="$(IFS=','; echo "${VALID_IDS[*]}")"
  echo " CLASSIFY_JOBIDS=${JOINED}"
  # Export for parent scripts to capture
  export CLASSIFY_JOBIDS_CSV="${JOINED}"
fi

echo "============================================================"
