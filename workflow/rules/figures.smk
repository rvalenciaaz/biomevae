# ──────────────────────────────────────────────────────────────────────
# figures.smk – Step 4: publication-quality figures for a single study.
#
# Wraps ``biomevae-single-study-figures``, which consumes every model's
# postprocess/classify outputs and writes fig1…fig5, the TSV/LaTeX
# summary, and confusion matrices to ``<output_root>/<study>/figures/``.
#
# Snakemake's DAG guarantees the figure rule only fires once all VAE
# classification runs AND the XGBoost baseline have completed.
# ──────────────────────────────────────────────────────────────────────


rule single_study_figures:
    input:
        classify = expand(
            OUTPUT_ROOT + "/{{study}}/models/{model}/classify/classification_results.json",
            model=ALL_MODELS,
        ),
        test_reports = expand(
            OUTPUT_ROOT + "/{{study}}/models/{model}/test/test_report.json",
            model=ALL_MODELS,
        ),
        baseline = (
            OUTPUT_ROOT
            + "/{study}/models/xgboost-baseline/classify/"
            + "xgboost_baseline_classification_results.json"
        ),
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        metadata  = lambda wc: data_path(wc.study, "sample_metadata.tsv"),
    output:
        summary = OUTPUT_ROOT + "/{study}/figures/results_summary.tsv",
    params:
        models_dir  = lambda wc: models_dir(wc.study),
        figures_dir = lambda wc: figures_dir(wc.study),
        label       = LABEL,
        study_name  = lambda wc: display_name(wc.study),
    log:
        OUTPUT_ROOT + "/{study}/figures/logs/figures.log",
    # HPC resources – mirror ``hpc/generate_figures.slurm``:
    # ei-short partition, 2h wall-time, 4 CPUs, 16G RAM, no GPU.
    threads: 4
    resources:
        partition   = "ei-medium",
        runtime     = 26 * 60,
        mem_mb      = 16 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p {params.figures_dir}/logs
        biomevae-single-study-figures \
            --results-dir {params.models_dir} \
            --metadata {input.metadata} \
            --outdir {params.figures_dir} \
            --label "{params.label}" \
            --study-name "{params.study_name}" \
            --input {input.sgb_table} \
            > {log} 2>&1
        """
