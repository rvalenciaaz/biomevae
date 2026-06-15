# ──────────────────────────────────────────────────────────────────────
# classify.smk – Step 3: classification on the chosen metadata label.
#
#   * classify              – runs ``biomevae-classify`` on every VAE's
#                             embeddings and writes classification
#                             metrics + confusion matrices under
#                             <model>/classify/.
#   * classify_xgboost_baseline – runs ``biomevae-classify-baseline`` (an
#                             XGBoost classifier trained directly on the
#                             raw SGB table) as a reference point for
#                             the VAE embeddings.
# ──────────────────────────────────────────────────────────────────────


rule classify:
    input:
        embeddings = OUTPUT_ROOT + "/{study}/models/{model}/embed/embeddings.tsv",
        metadata   = lambda wc: data_path(wc.study, "sample_metadata.tsv"),
    output:
        results = OUTPUT_ROOT + "/{study}/models/{model}/classify/classification_results.json",
    params:
        classify_dir = lambda wc: os.path.join(model_dir(wc.study, wc.model), "classify"),
        label        = LABEL,
        seeds        = EVAL_SEEDS_STR,
    log:
        OUTPUT_ROOT + "/{study}/models/{model}/logs/classify.log",
    # HPC resources – mirror ``hpc/classify_model.slurm``:
    # ei-short partition, 2h wall-time, 4 CPUs, 16G RAM, no GPU.
    threads: 4
    resources:
        partition   = "ei-medium",
        runtime     = 26 * 60,
        mem_mb      = 16 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p {params.classify_dir}
        biomevae-classify \
            --embeddings {input.embeddings} \
            --metadata {input.metadata} \
            --label {params.label} \
            --outdir {params.classify_dir} \
            --n-splits 5 \
            --n-repeats 10 \
            --seeds {params.seeds} \
            > {log} 2>&1
        """


rule classify_xgboost_baseline:
    input:
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        metadata  = lambda wc: data_path(wc.study, "sample_metadata.tsv"),
    output:
        results = (
            OUTPUT_ROOT
            + "/{study}/models/xgboost-baseline/classify/"
            + "xgboost_baseline_classification_results.json"
        ),
    params:
        baseline_dir = lambda wc: os.path.join(
            models_dir(wc.study), "xgboost-baseline", "classify"
        ),
        label = LABEL,
        seeds = EVAL_SEEDS_STR,
    log:
        OUTPUT_ROOT + "/{study}/models/xgboost-baseline/logs/classify.log",
    # HPC resources – mirror ``hpc/classify_xgboost_baseline.slurm``.
    threads: 4
    resources:
        partition   = "ei-medium",
        runtime     = 26 * 60,
        mem_mb      = 16 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p {params.baseline_dir}
        mkdir -p $(dirname {log})
        biomevae-classify-baseline \
            --input {input.sgb_table} \
            --metadata {input.metadata} \
            --label {params.label} \
            --outdir {params.baseline_dir} \
            --log1p \
            --n-splits 5 \
            --n-repeats 10 \
            --seeds {params.seeds} \
            > {log} 2>&1
        """
