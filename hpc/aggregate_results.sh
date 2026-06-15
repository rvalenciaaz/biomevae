#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# aggregate_results.sh – collect results across all models after
#                        post-processing and generate summary artifacts
#
# Usage (called by aggregate_results.slurm via srun):
#   ./aggregate_results.sh <OUTDIR> <INPUT> [TAXONOMY]
#
# Arguments:
#   OUTDIR    – root output directory (contains per-model sub-folders)
#   INPUT     – path to the abundance table (for benchmark figures)
#   TAXONOMY  – path to taxonomy file (pass "none" to skip)
#
# Steps executed:
#   1. Collect test_report.json from each model into a summary TSV
#   2. Plot training curves across all models
#   3. (Optional) Generate benchmark comparison figure if enough data
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

OUTDIR="${1:?Usage: $0 <OUTDIR> <INPUT> [TAXONOMY]}"
INPUT="${2:?Missing INPUT path}"
TAXONOMY="${3:-none}"

# ── evaluation seeds ─────────────────────────────────────────────────
# Every downstream reconstruction / classification / enterosignature
# evaluation is pooled across these seeds for reproducibility. Kept in
# sync with ``biomevae.classify.DEFAULT_EVAL_SEEDS``. Override via the
# ``EVAL_SEEDS`` environment variable, e.g. ``EVAL_SEEDS="42 43" ...``.
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
read -r -a EVAL_SEEDS_ARRAY <<< "${EVAL_SEEDS}"

# ── logging ──────────────────────────────────────────────────────────
LOG_DIR="${OUTDIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/aggregate_$(date +'%Y%m%d_%H%M%S').log"

log_with_timestamp() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "Starting aggregation for: ${OUTDIR}"

# ── conda / mamba environment ────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/hpc-home/her24bip/.local/share/mamba}"
MAMBA_EXEC="${MAMBA_EXEC:-/hpc-home/her24bip/miniconda3/condabin/mamba}"
CONDA_ENV="${CONDA_ENV:-biomevae}"

AGGREGATE_DIR="${OUTDIR}/aggregate"
mkdir -p "${AGGREGATE_DIR}"

# ════════════════════════════════════════════════════════════════════
# Step 1: Collect test_report.json into summary TSV
# ════════════════════════════════════════════════════════════════════
log_with_timestamp "Collecting test reports..."

SUMMARY_FILE="${AGGREGATE_DIR}/test_summary.tsv"
# Use the UNION of keys across test_report.json files rather than requiring
# identical schemas.  Different model families (tree-based NB-NLL models,
# euclid/hyperbolic β-VAEs, PhILR, …) can legitimately emit overlapping-but-
# not-identical metric sets; missing keys are written as empty cells.  Also
# guarantee that the output TSV exists even if no reports are found, so the
# Snakemake ``aggregate`` rule's output check is satisfied.
if "${MAMBA_EXEC}" run -n "${CONDA_ENV}" python - "$OUTDIR" "$SUMMARY_FILE" <<'PY'
import json
import pathlib
import sys

outdir = pathlib.Path(sys.argv[1])
summary_file = pathlib.Path(sys.argv[2])
summary_file.parent.mkdir(parents=True, exist_ok=True)

rows = []
all_keys: set[str] = set()

for model_dir in sorted(p for p in outdir.iterdir() if p.is_dir()):
    report = model_dir / "test" / "test_report.json"
    if not report.exists():
        continue
    try:
        with report.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"WARNING: failed to parse {report}: {exc}", file=sys.stderr)
        continue
    if not isinstance(payload, dict):
        print(f"WARNING: skipping non-object test report {report}", file=sys.stderr)
        continue
    all_keys.update(str(k) for k in payload.keys())
    rows.append((model_dir.name, payload))

header = sorted(all_keys)
with summary_file.open("w", encoding="utf-8") as out:
    out.write("model\t" + "\t".join(header) + "\n")
    for name, payload in rows:
        out.write(
            name
            + "\t"
            + "\t".join("" if k not in payload else str(payload[k]) for k in header)
            + "\n"
        )

if not rows:
    # Emit a non-zero status so the calling shell logs a WARNING, but the
    # (header-only) file has already been written above to satisfy Snakemake.
    raise SystemExit(2)
PY
then
  log_with_timestamp "Saved test summary to: ${SUMMARY_FILE}"
else
  if [[ -s "${SUMMARY_FILE}" ]]; then
    log_with_timestamp "WARNING: Test summary written with partial/empty rows."
  else
    log_with_timestamp "WARNING: No test reports found to aggregate; wrote header-only TSV."
  fi
fi

# Last-resort guarantee: the Snakemake rule declares ``test_summary.tsv`` as
# a hard output.  If for any reason the python block above could not write
# the file (e.g. I/O error, mamba wrapper glitch), drop an empty stub so the
# DAG is not broken.
if [[ ! -f "${SUMMARY_FILE}" ]]; then
  printf "model\n" > "${SUMMARY_FILE}"
  log_with_timestamp "NOTE: wrote placeholder ${SUMMARY_FILE}"
fi

# ════════════════════════════════════════════════════════════════════
# Step 2: Plot training curves across all models
# ════════════════════════════════════════════════════════════════════
log_with_timestamp "Plotting training curves..."

LOG_ARGS=()
for model_dir in "${OUTDIR}"/*/; do
  model_name="$(basename "${model_dir}")"
  train_log="${model_dir}training_log.tsv"
  if [[ -f "${train_log}" ]]; then
    LOG_ARGS+=(--log "${model_name}=${train_log}")
  fi
done

if [[ ${#LOG_ARGS[@]} -gt 0 ]]; then
  # Plot both the β-weighted ELBO (``loss``) and the stationary
  # reconstruction signal (``recon``). The recon curve is the only
  # convergence diagnostic that is comparable across epochs for
  # β-annealed VAEs and across models for NB-NLL based architectures
  # (PhILR-VAE, TreeNB-VAE), where plotting only the ELBO would leave
  # them without a stationary curve.
  for metric in loss recon; do
    CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-plot-training-curves
      "${LOG_ARGS[@]}"
      --metric "${metric}"
      --output "${AGGREGATE_DIR}"
      --title "All models"
    )

    log_with_timestamp "Running: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
      log_with_timestamp "Training ${metric} curves saved to: ${AGGREGATE_DIR}"
    else
      log_with_timestamp "WARNING: biomevae-plot-training-curves (--metric ${metric}) failed."
    fi
  done
else
  log_with_timestamp "WARNING: No training_log.tsv files found."
fi

# ════════════════════════════════════════════════════════════════════
# Step 3: Collect embeddings info for benchmark figure (optional)
# ════════════════════════════════════════════════════════════════════
log_with_timestamp "Checking for embedding files for ordination plots..."

EMBED_ARGS=()
for model_dir in "${OUTDIR}"/*/; do
  model_name="$(basename "${model_dir}")"
  emb_file=""
  for candidate in "${model_dir}embed/embeddings.tsv" "${model_dir}test/embeddings.tsv" "${model_dir}embeddings.tsv"; do
    if [[ -f "${candidate}" ]]; then
      emb_file="${candidate}"
      break
    fi
  done
  if [[ -n "${emb_file}" ]]; then
    EMBED_ARGS+=(--embedding "${model_name}=${emb_file}")
    log_with_timestamp "  Found embeddings: ${model_name}"
  fi
done

HAS_COMPARATIVE_EMBEDS=false
if [[ ${#EMBED_ARGS[@]} -lt 2 ]]; then
  log_with_timestamp "Fewer than 2 embedding files found; comparative embedding figures will be skipped."
else
  HAS_COMPARATIVE_EMBEDS=true
  log_with_timestamp "Found ${#EMBED_ARGS[@]} embedding files."
fi

# ════════════════════════════════════════════════════════════════════
# Step 4: biomevae-allcomp – cross-validation benchmark comparison
# ════════════════════════════════════════════════════════════════════
log_with_timestamp "Running cross-validation benchmark comparison..."

METHOD_ARGS=()
for model_dir in "${OUTDIR}"/*/; do
  model_name="$(basename "${model_dir}")"
  config_file="${model_dir}config.json"
  if [[ -f "${config_file}" ]]; then
    METHOD_ARGS+=(--method "${model_name}=${config_file}")
    log_with_timestamp "  Found config: ${model_name}"
  fi
done

ALLCOMP_JSON="${AGGREGATE_DIR}/all_methods_vs_nmf.json"

# Aggregate runs on the CPU-only ``ei-short`` partition, so force every
# method onto the CPU regardless of what the model's ``config.json`` stored
# (typically ``cuda`` from the training run).  Without this override, models
# that were trained on GPU would fail to move to the requested device during
# the cross-validation re-training inside biomevae-allcomp.
ALLCOMP_DEVICE="${ALLCOMP_DEVICE:-cpu}"
ALLCOMP_OK=false

if [[ ${#METHOD_ARGS[@]} -ge 2 ]]; then
  TAX_FLAG=()
  if [[ "${TAXONOMY}" != "none" ]]; then
    TAX_FLAG=(--taxonomy "${TAXONOMY}")
  fi

  CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-allcomp
    --input "${INPUT}"
    --components 16
    --splits 5
    --seeds "${EVAL_SEEDS_ARRAY[@]}"
    --device "${ALLCOMP_DEVICE}"
    "${METHOD_ARGS[@]}"
    "${TAX_FLAG[@]}"
    --output "${ALLCOMP_JSON}"
  )

  log_with_timestamp "Running: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "biomevae-allcomp completed. Output: ${ALLCOMP_JSON}"
    if [[ -s "${ALLCOMP_JSON}" ]]; then
      ALLCOMP_OK=true
    fi
  else
    log_with_timestamp "WARNING: biomevae-allcomp failed."
  fi
else
  log_with_timestamp "WARNING: Fewer than 2 model configs found; skipping biomevae-allcomp."
fi

# The Snakemake aggregate rule declares ``all_methods_vs_nmf.json`` as a hard
# output.  When biomevae-allcomp cannot run to completion (e.g. the ei-short
# partition times out re-training 10 models × 5 seeds × 5 splits on CPU, or
# a single method raises during CV) we fall back to an empty JSON object so
# the DAG is not broken and downstream figure steps degrade gracefully
# (``biomevae-benchmark-figure`` et al. already guard on an empty/missing
# allcomp payload via their ``[[ -f "${ALLCOMP_JSON}" ]]`` check above).
if [[ ! -f "${ALLCOMP_JSON}" ]]; then
  printf "{}\n" > "${ALLCOMP_JSON}"
  log_with_timestamp "NOTE: wrote placeholder ${ALLCOMP_JSON}"
fi

# ════════════════════════════════════════════════════════════════════
# Step 5: biomevae-benchmark-figure – metric bar charts + ordinations
# ════════════════════════════════════════════════════════════════════
FIGURES_DIR="${AGGREGATE_DIR}/figures"
mkdir -p "${FIGURES_DIR}"

if [[ "${ALLCOMP_OK}" == "true" ]]; then
  log_with_timestamp "Generating benchmark figures..."

  if [[ "${HAS_COMPARATIVE_EMBEDS}" == "true" ]]; then
    CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-benchmark-figure
      --input "${ALLCOMP_JSON}"
      --metric rmse --metric mae --metric r2 --metric r2_per_feature
      --title "Reconstruction benchmark"
      --baseline nmf
      --matrix "${INPUT}"
      "${EMBED_ARGS[@]}"
      --output "${FIGURES_DIR}/benchmark_metrics.pdf"
      --ordinations-output "${FIGURES_DIR}/benchmark_ordinations.pdf"
    )

    log_with_timestamp "Running: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
      log_with_timestamp "Benchmark figures saved to: ${FIGURES_DIR}"
    else
      log_with_timestamp "WARNING: biomevae-benchmark-figure failed."
    fi
  else
    log_with_timestamp "Skipping biomevae-benchmark-figure: requires at least 2 embedding files."
  fi

  # ── Step 5b: Reconstruction violin plots ──────────────────────────
  CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-recon-violin
    --input "${ALLCOMP_JSON}"
    --metric rmse --metric mae --metric r2
    --baseline nmf
    --output "${FIGURES_DIR}/benchmark_violin.pdf"
  )

  log_with_timestamp "Running: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "Violin plots saved."
  else
    log_with_timestamp "WARNING: biomevae-recon-violin failed."
  fi

  # ── Step 5c: Pairwise significance tables ─────────────────────────
  CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-pairwise-table
    --input "${ALLCOMP_JSON}"
    --metric rmse --metric mae
    --output "${FIGURES_DIR}/pairwise"
    --format both
  )

  log_with_timestamp "Running: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "Pairwise tables saved."
  else
    log_with_timestamp "WARNING: biomevae-pairwise-table failed."
  fi

  # ── Step 5d: Hierarchy figure (if taxonomy provided) ──────────────
  if [[ "${TAXONOMY}" != "none" ]]; then
    CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-hierarchy-figure
      --input "${ALLCOMP_JSON}"
      --metric rmse --metric mae
      --baseline nmf
      --output "${FIGURES_DIR}/hierarchy_metrics.pdf"
    )

    log_with_timestamp "Running: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
      log_with_timestamp "Hierarchy figures saved."
    else
      log_with_timestamp "WARNING: biomevae-hierarchy-figure failed."
    fi
  fi

  # ── Step 5e: Reconstruction scatter plots ─────────────────────────
  RECON_ARGS=()
  for model_dir in "${OUTDIR}"/*/; do
    model_name="$(basename "${model_dir}")"
    recon_file=""
    for candidate in "${model_dir}embed/recon.tsv" "${model_dir}test/recon.tsv" "${model_dir}recon.tsv"; do
      if [[ -f "${candidate}" ]]; then
        recon_file="${candidate}"
        break
      fi
    done
    if [[ -n "${recon_file}" ]]; then
      RECON_ARGS+=(--recon "${model_name}=${recon_file}")
    fi
  done

  if [[ ${#RECON_ARGS[@]} -ge 1 ]]; then
    CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-recon-scatter
      --input "${INPUT}"
      "${RECON_ARGS[@]}"
      --output "${FIGURES_DIR}/recon_scatter.pdf"
    )

    log_with_timestamp "Running: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
      log_with_timestamp "Scatter plots saved."
    else
      log_with_timestamp "WARNING: biomevae-recon-scatter failed."
    fi
  fi
else
  log_with_timestamp "WARNING: No usable allcomp JSON; skipping benchmark figures."
fi

# ════════════════════════════════════════════════════════════════════
# Step 6: biomevae-benchmark-figures-enterosignatures
# ════════════════════════════════════════════════════════════════════
if [[ "${ALLCOMP_OK}" == "true" && "${TAXONOMY}" != "none" && "${HAS_COMPARATIVE_EMBEDS}" == "true" ]]; then
  log_with_timestamp "Generating enterosignature figures..."

  ENTERO_DIR="${FIGURES_DIR}/enterosignatures"
  mkdir -p "${ENTERO_DIR}"

  CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-benchmark-figures-enterosignatures
    --input "${ALLCOMP_JSON}"
    --metric rmse --metric mae
    --title "Reconstruction benchmark"
    --baseline nmf
    --matrix "${INPUT}"
    --taxonomy "${TAXONOMY}"
    "${EMBED_ARGS[@]}"
    --clusters 2
    --seeds "${EVAL_SEEDS_ARRAY[@]}"
    --output "${ENTERO_DIR}/benchmark_metrics.pdf"
    --ordinations-output "${ENTERO_DIR}/benchmark_ordinations.pdf"
    --enterosignature-output "${ENTERO_DIR}/enterosignatures.pdf"
    --comparison-output "${ENTERO_DIR}/enterosignature_agreement.pdf"
    --agreement-output "${ENTERO_DIR}/enterosignature_ari.pdf"
    --geometry-plot-output "${ENTERO_DIR}/enterosignature_geometry.pdf"
    --procrustes-output "${ENTERO_DIR}/enterosignature_procrustes.pdf"
    --contingency-plot-output "${ENTERO_DIR}/enterosignature_contingency.pdf"
  )

  log_with_timestamp "Running: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "Enterosignature figures saved to: ${ENTERO_DIR}"
  else
    log_with_timestamp "WARNING: biomevae-benchmark-figures-enterosignatures failed."
  fi
else
  log_with_timestamp "Skipping enterosignature figures (requires allcomp JSON + taxonomy + >=2 embeddings)."
fi

# ════════════════════════════════════════════════════════════════════
# Step 7: biomevae-benchmark-slides – LaTeX Beamer deck
# ════════════════════════════════════════════════════════════════════
if [[ "${ALLCOMP_OK}" == "true" && "${HAS_COMPARATIVE_EMBEDS}" == "true" ]]; then
  log_with_timestamp "Generating benchmark slides..."

  SLIDES_DIR="${AGGREGATE_DIR}/slides"
  mkdir -p "${SLIDES_DIR}"

  CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-benchmark-slides
    --input "${ALLCOMP_JSON}"
    --metric rmse --metric mae
    --title "Benchmark Overview"
    --subtitle "Hold-out reconstruction"
    --author "biomevae"
    --matrix "${INPUT}"
    "${EMBED_ARGS[@]}"
    --figure-output "${SLIDES_DIR}/benchmark_figure.pdf"
    --ordinations-output "${SLIDES_DIR}/benchmark_ordinations.pdf"
    --slides-output "${SLIDES_DIR}/benchmark_slides.tex"
  )

  log_with_timestamp "Running: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "Slides saved to: ${SLIDES_DIR}"
  else
    log_with_timestamp "WARNING: biomevae-benchmark-slides failed."
  fi
else
  log_with_timestamp "Skipping benchmark slides (requires allcomp JSON + >=2 embeddings)."
fi

# ════════════════════════════════════════════════════════════════════
# Step 8: biomevae-interpret-compare – cross-model SHAP comparison
# ════════════════════════════════════════════════════════════════════
log_with_timestamp "Checking for interpretation directories..."

INTERP_ARGS=()
for model_dir in "${OUTDIR}"/*/; do
  model_name="$(basename "${model_dir}")"
  interp_dir="${model_dir}interpret"
  if [[ -d "${interp_dir}" && -f "${interp_dir}/otu_latent_summary.tsv" ]]; then
    INTERP_ARGS+=(--interpret-dir "${model_name}=${interp_dir}")
    log_with_timestamp "  Found interpret dir: ${model_name}"
  fi
done

if [[ ${#INTERP_ARGS[@]} -ge 2 ]]; then
  CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-interpret-compare
    "${INTERP_ARGS[@]}"
    --top-k 20
    --output "${FIGURES_DIR}/interpret_comparison"
  )

  log_with_timestamp "Running: ${CMD[*]}"
  if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    log_with_timestamp "Interpretation comparison saved."
  else
    log_with_timestamp "WARNING: biomevae-interpret-compare failed."
  fi
else
  log_with_timestamp "Fewer than 2 interpret directories found; skipping comparison."
fi

# ════════════════════════════════════════════════════════════════════
# Final guarantee: every aggregate figure that the Snakemake
# ``aggregate`` rule declares as an output must exist on disk by the
# time this script returns.
#
# When the corresponding step succeeded above, the real PDF is already
# present and the ``touch`` below is a no-op.  When it failed (for
# example: fewer than two models converged, the CPU-only partition
# could not finish ``biomevae-allcomp`` within the wall-time, or the
# taxonomy was disabled), we drop a zero-byte placeholder so the
# downstream DAG is not broken — mirroring how
# ``all_methods_vs_nmf.json`` and ``test_summary.tsv`` are stubbed
# earlier in this script.  The placeholders are obviously empty, so a
# user inspecting the figures directory can tell at a glance which
# benchmark step actually ran.
log_with_timestamp "Verifying aggregate figure outputs..."

mkdir -p "${FIGURES_DIR}"
for stub in \
    "${FIGURES_DIR}/benchmark_metrics.pdf" \
    "${FIGURES_DIR}/benchmark_ordinations.pdf" \
    "${FIGURES_DIR}/benchmark_violin.pdf" \
    "${FIGURES_DIR}/hierarchy_metrics.pdf" \
    "${FIGURES_DIR}/recon_scatter.pdf"; do
  if [[ ! -e "${stub}" ]]; then
    : > "${stub}"
    log_with_timestamp "NOTE: wrote placeholder $(basename "${stub}")"
  fi
done

log_with_timestamp "Aggregation completed."
exit 0
