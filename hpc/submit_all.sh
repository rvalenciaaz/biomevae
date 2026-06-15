#!/usr/bin/env bash
set -euo pipefail

INPUT=""
TAXONOMY=""
OUTDIR=""
EXTRA_ARGS=""
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 --input <FILE> --outdir <DIR> [OPTIONS]

Required:
  --input      Path to the abundance table (TSV/CSV)
  --taxonomy   Path to the taxonomy file (TSV/CSV). Required only for
               taxonomy-aware models; those jobs are skipped if omitted.
  --outdir     Root output directory (a sub-folder per model is created)

Optional:
  --extra-args   Additional CLI flags passed to every model training command
                 (quote the whole string, e.g. "--epochs 500 --optuna").
                 For the supervised DS-VAE variant (dsvae-sup) you will
                 typically include "--metadata <path> --label-col disease".
  --dry-run      Print sbatch commands without submitting
  -h, --help     Show this help message
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)      INPUT="$2";      shift 2 ;;
    --taxonomy)   TAXONOMY="$2";   shift 2 ;;
    --outdir)     OUTDIR="$2";     shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=true;    shift   ;;
    -h|--help)    usage 0 ;;
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
SLURM_SCRIPT="${SCRIPT_DIR}/train_model.slurm"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_model.sh"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Error: cannot find ${SLURM_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: cannot find ${TRAIN_SCRIPT}" >&2
  exit 1
fi

MODELS=(
  "biomevae-train|beta-vae|no|"
  "biomevae-train-vanilla|vanilla-vae|no|"
  "biomevae-train-hyp|hyp-vae|no|"
  "biomevae-train-tax|tax-vae|yes|"
  "biomevae-train-hyp-tax|hyp-tax-vae|yes|"
  "biomevae-train-graph|graph-vae|yes|"
  "biomevae-train-treeprior|treeprior-vae|yes|"
  "biomevae-train-fuse|fuse-vae|yes|"
  "biomevae-train-tree-dtm|tree-dtm-vae|yes|"
  "biomevae-train-philrvae|philrvae|yes|"
  "biomevae-train-hyp-philrvae|hyp-philrvae|yes|"
  "biomevae-train-hyp-philr-zinb|hyp-philr-zinb|yes|"
  # DS-VAE variants share a single CLI; supervised variant needs
  # --supervised (plus the caller must pass --metadata/--label-col
  # through --extra-args when submitting the supervised run).
  "biomevae-train-dsvae|dsvae-unsup|yes|--no-supervised"
  "biomevae-train-dsvae|dsvae-sup|yes|--supervised"
  # Single-study CAPDA-VAE: taxonomy-aware and metadata-aware (the caller
  # must pass --metadata/--label-col through --extra-args, as for dsvae-sup).
  "biomevae-train-capda-vae-ss|capda-vae|yes|"
)

echo "============================================================"
echo " biomevae – parallel HPC training submission"
echo "============================================================"
echo " INPUT    : ${INPUT}"
echo " TAXONOMY : ${TAXONOMY}"
echo " OUTDIR   : ${OUTDIR}"
echo " EXTRA    : ${EXTRA_ARGS:-<none>}"
echo " DRY RUN  : ${DRY_RUN}"
echo "------------------------------------------------------------"

EXTRA_ARGS_B64="$(printf '%s' "${EXTRA_ARGS}" | base64 -w0)"

SUBMITTED=0
SKIPPED=0

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_CMD JOB_NAME NEEDS_TAX MODEL_FLAGS <<< "$entry"

  MODEL_OUTDIR="${OUTDIR}/${JOB_NAME}"
  mkdir -p "${MODEL_OUTDIR}"

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

  # Per-model CLI flags (e.g. --supervised for DS-VAE) are prepended
  # to the user-supplied --extra-args so they reach the training CLI
  # through the same base64 channel used by train_model.sh.
  if [[ -n "${MODEL_FLAGS:-}" ]]; then
    PER_MODEL_EXTRA="${MODEL_FLAGS} ${EXTRA_ARGS}"
  else
    PER_MODEL_EXTRA="${EXTRA_ARGS}"
  fi
  PER_MODEL_EXTRA_B64="$(printf '%s' "${PER_MODEL_EXTRA}" | base64 -w0)"

  SBATCH_CMD=(
    sbatch
    --job-name="${JOB_NAME}"
    --output="${MODEL_OUTDIR}/slurm_%j.out"
    --error="${MODEL_OUTDIR}/slurm_%j.err"
    --export="ALL,MODEL_CMD=${MODEL_CMD},INPUT=${INPUT},OUTDIR=${MODEL_OUTDIR},TAXONOMY=${TAX_ARG},EXTRA_ARGS_B64=${PER_MODEL_EXTRA_B64},TRAIN_SCRIPT=${TRAIN_SCRIPT}"
    "${SLURM_SCRIPT}"
  )

  if $DRY_RUN; then
    echo "[DRY-RUN] ${SBATCH_CMD[*]}"
  else
    echo -n "Submitting ${JOB_NAME} (${MODEL_CMD})… "
    JOB_ID=$("${SBATCH_CMD[@]}" 2>&1)
    echo "${JOB_ID}"
  fi

  SUBMITTED=$((SUBMITTED + 1))
done

echo "------------------------------------------------------------"
echo " ${SUBMITTED} jobs submitted, ${SKIPPED} skipped."
echo "============================================================"
