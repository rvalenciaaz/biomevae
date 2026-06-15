#!/usr/bin/env bash
set -euo pipefail

INPUT=""
TAXONOMY=""
OUTDIR=""
EXTRA_ARGS=""
DRY_RUN=false
TRAIN_JOBIDS=""

usage() {
  cat <<EOF
Usage: $0 --input <FILE> --taxonomy <FILE> --outdir <DIR> [OPTIONS]

Submit post-processing (test, embed, interpret) for all trained biomevae models.

Required:
  --input      Path to the abundance table (TSV/CSV)
  --taxonomy   Path to the taxonomy file (TSV/CSV). Required only for
               taxonomy-aware models; those jobs are skipped if omitted.
  --outdir     Root output directory (must match the --outdir used for training)

Optional:
  --extra-args     Additional CLI flags passed to post-processing commands
                   (quote the whole string, e.g. "--device cpu --top-k 20")
  --train-jobids   Comma-separated SLURM job IDs from training submission.
                   Post-processing jobs will depend on the corresponding
                   training job via --dependency=afterok:<jobid>.
                   Order must match the model list (beta-vae, vanilla-vae,
                   hyp-vae, tax-vae, hyp-tax-vae, graph-vae, treeprior-vae,
                   fuse-vae, tree-dtm-vae, philrvae, hyp-philrvae,
                   hyp-philr-zinb).
  --dry-run        Print sbatch commands without submitting
  -h, --help       Show this help message
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)        INPUT="$2";        shift 2 ;;
    --taxonomy)     TAXONOMY="$2";     shift 2 ;;
    --outdir)       OUTDIR="$2";       shift 2 ;;
    --extra-args)   EXTRA_ARGS="$2";   shift 2 ;;
    --train-jobids) TRAIN_JOBIDS="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true;      shift   ;;
    -h|--help)      usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ -z "$INPUT"    ]] && { echo "Error: --input is required."    >&2; usage 1; }
[[ -z "$OUTDIR"   ]] && { echo "Error: --outdir is required."   >&2; usage 1; }
[[ ! -f "$INPUT"  ]] && { echo "Error: --input file not found: $INPUT" >&2; exit 1; }
if [[ -n "$TAXONOMY" ]] && [[ ! -f "$TAXONOMY" ]]; then
  echo "Error: --taxonomy file not found: $TAXONOMY" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/postprocess_model.slurm"
PP_SCRIPT="${SCRIPT_DIR}/postprocess_model.sh"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Error: cannot find ${SLURM_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "$PP_SCRIPT" ]]; then
  echo "Error: cannot find ${PP_SCRIPT}" >&2
  exit 1
fi

# Same model list as submit_all.sh: MODEL_CMD|JOB_NAME|NEEDS_TAX
MODELS=(
  "biomevae-train|beta-vae|no"
  "biomevae-train-vanilla|vanilla-vae|no"
  "biomevae-train-hyp|hyp-vae|no"
  "biomevae-train-tax|tax-vae|yes"
  "biomevae-train-hyp-tax|hyp-tax-vae|yes"
  "biomevae-train-graph|graph-vae|yes"
  "biomevae-train-treeprior|treeprior-vae|yes"
  "biomevae-train-fuse|fuse-vae|yes"
  "biomevae-train-tree-dtm|tree-dtm-vae|yes"
  "biomevae-train-philrvae|philrvae|yes"
  "biomevae-train-hyp-philrvae|hyp-philrvae|yes"
  "biomevae-train-hyp-philr-zinb|hyp-philr-zinb|yes"
)

# Parse --train-jobids into an array (empty if not provided)
IFS=',' read -ra JOBID_ARRAY <<< "${TRAIN_JOBIDS}"
EXTRA_ARGS_B64="$(printf '%s' "${EXTRA_ARGS}" | base64 -w0)"
if [[ -n "${TRAIN_JOBIDS}" ]] && [[ ${#JOBID_ARRAY[@]} -ne ${#MODELS[@]} ]]; then
  echo "Error: --train-jobids must contain exactly ${#MODELS[@]} comma-separated IDs (one per model)." >&2
  exit 1
fi

echo "============================================================"
echo " biomevae – parallel HPC post-processing submission"
echo "============================================================"
echo " INPUT    : ${INPUT}"
echo " TAXONOMY : ${TAXONOMY}"
echo " OUTDIR   : ${OUTDIR}"
echo " EXTRA    : ${EXTRA_ARGS:-<none>}"
echo " JOBIDS   : ${TRAIN_JOBIDS:-<none>}"
echo " DRY RUN  : ${DRY_RUN}"
echo "------------------------------------------------------------"

SUBMITTED=0
SKIPPED=0

for i in "${!MODELS[@]}"; do
  entry="${MODELS[$i]}"
  IFS='|' read -r _MODEL_CMD JOB_NAME NEEDS_TAX <<< "$entry"

  MODEL_OUTDIR="${OUTDIR}/${JOB_NAME}"

  # If no training dependency was given, check that artifacts exist
  if [[ -z "${TRAIN_JOBIDS}" ]]; then
    if [[ ! -f "${MODEL_OUTDIR}/model.pt" ]] || [[ ! -f "${MODEL_OUTDIR}/config.json" ]]; then
      echo "SKIP ${JOB_NAME}: model.pt or config.json not found in ${MODEL_OUTDIR}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
  fi

  if [[ "$NEEDS_TAX" == "yes" ]]; then
    if [[ -z "$TAXONOMY" ]]; then
      echo "Skipping ${JOB_NAME}: taxonomy-aware model requires --taxonomy."
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    TAX_ARG="${TAXONOMY}"
  else
    TAX_ARG="none"
  fi

  # Build sbatch command
  SBATCH_CMD=(
    sbatch
    --job-name="pp-${JOB_NAME}"
    --output="${MODEL_OUTDIR}/slurm_pp_%j.out"
    --error="${MODEL_OUTDIR}/slurm_pp_%j.err"
    --export="ALL,MODEL_OUTDIR=${MODEL_OUTDIR},INPUT=${INPUT},TAXONOMY=${TAX_ARG},EXTRA_ARGS_B64=${EXTRA_ARGS_B64},PP_SCRIPT=${PP_SCRIPT}"
  )

  # Add SLURM dependency if a training job ID was provided for this model
  if [[ ${#JOBID_ARRAY[@]} -gt 0 ]] && [[ -n "${JOBID_ARRAY[$i]:-}" ]]; then
    SBATCH_CMD+=(--dependency="afterok:${JOBID_ARRAY[$i]}")
  fi

  SBATCH_CMD+=("${SLURM_SCRIPT}")

  if $DRY_RUN; then
    echo "[DRY-RUN] ${SBATCH_CMD[*]}"
  else
    mkdir -p "${MODEL_OUTDIR}"
    echo -n "Submitting pp-${JOB_NAME}… "
    JOB_ID=$("${SBATCH_CMD[@]}" 2>&1)
    echo "${JOB_ID}"
  fi

  SUBMITTED=$((SUBMITTED + 1))
done

echo "------------------------------------------------------------"
echo " ${SUBMITTED} jobs submitted, ${SKIPPED} skipped."
echo "============================================================"
