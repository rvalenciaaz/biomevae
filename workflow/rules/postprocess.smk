# ──────────────────────────────────────────────────────────────────────
# postprocess.smk – Step 2: per-model post-training analysis.
#
# Mirrors the four steps of ``hpc/postprocess_model.sh``:
#   * postprocess_test          – biomevae-test  --export  (metrics, recon,
#                                 embeddings under <model>/test/)
#   * postprocess_embed         – biomevae-embed (fresh embeddings and
#                                 reconstruction under <model>/embed/)
#   * postprocess_interpret     – biomevae-interpret (SHAP attributions
#                                 under <model>/interpret/)
#   * postprocess_interpret_genus – biomevae-interpret aggregated to genus
#                                 level under <model>/interpret_genus/
#
# Every rule takes the training artifacts (``model.pt`` + ``config.json``)
# as input so Snakemake wires the DAG correctly.
# ──────────────────────────────────────────────────────────────────────


def _pp_tax_flag(wildcards, input):
    if MODELS[wildcards.model]["needs_tax"]:
        return f"--taxonomy {input.taxonomy}"
    return ""


rule postprocess_test:
    input:
        model_pt  = OUTPUT_ROOT + "/{study}/models/{model}/model.pt",
        config    = OUTPUT_ROOT + "/{study}/models/{model}/config.json",
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        taxonomy  = lambda wc: data_path(wc.study, "phyla.tsv"),
    output:
        report     = OUTPUT_ROOT + "/{study}/models/{model}/test/test_report.json",
        embeddings = OUTPUT_ROOT + "/{study}/models/{model}/test/embeddings.tsv",
    params:
        model_dir = lambda wc: model_dir(wc.study, wc.model),
        test_dir  = lambda wc: os.path.join(model_dir(wc.study, wc.model), "test"),
        tax_flag  = _pp_tax_flag,
    log:
        OUTPUT_ROOT + "/{study}/models/{model}/logs/postprocess_test.log",
    # HPC resources – mirror ``hpc/postprocess_model.slurm``:
    # ei-gpu partition, 24h wall-time, 20 CPUs, 128G RAM, 1 GPU.
    threads: 20
    resources:
        partition   = "ei-gpu",
        runtime     = 48 * 60,
        mem_mb      = 128 * 1024,
        gpus        = 1,
        slurm_extra = "--gres=gpu:1",
    shell:
        r"""
        mkdir -p {params.test_dir}
        biomevae-test \
            --input {input.sgb_table} \
            --model-dir {params.model_dir} \
            --outdir {params.test_dir} \
            --export \
            {params.tax_flag} \
            > {log} 2>&1
        """


rule postprocess_embed:
    input:
        model_pt  = OUTPUT_ROOT + "/{study}/models/{model}/model.pt",
        config    = OUTPUT_ROOT + "/{study}/models/{model}/config.json",
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        taxonomy  = lambda wc: data_path(wc.study, "phyla.tsv"),
    output:
        embeddings = OUTPUT_ROOT + "/{study}/models/{model}/embed/embeddings.tsv",
        recon      = OUTPUT_ROOT + "/{study}/models/{model}/embed/recon.tsv",
    params:
        model_dir = lambda wc: model_dir(wc.study, wc.model),
        embed_dir = lambda wc: os.path.join(model_dir(wc.study, wc.model), "embed"),
        tax_flag  = _pp_tax_flag,
    log:
        OUTPUT_ROOT + "/{study}/models/{model}/logs/postprocess_embed.log",
    threads: 20
    resources:
        partition   = "ei-gpu",
        runtime     = 48 * 60,
        mem_mb      = 128 * 1024,
        gpus        = 1,
        slurm_extra = "--gres=gpu:1",
    shell:
        r"""
        mkdir -p {params.embed_dir}
        biomevae-embed \
            --input {input.sgb_table} \
            --model-dir {params.model_dir} \
            --outdir {params.embed_dir} \
            --export-recon \
            {params.tax_flag} \
            > {log} 2>&1
        """


rule postprocess_interpret:
    input:
        model_pt  = OUTPUT_ROOT + "/{study}/models/{model}/model.pt",
        config    = OUTPUT_ROOT + "/{study}/models/{model}/config.json",
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        taxonomy  = lambda wc: data_path(wc.study, "phyla.tsv"),
    output:
        summary = OUTPUT_ROOT + "/{study}/models/{model}/interpret/otu_latent_summary.tsv",
    params:
        model_dir     = lambda wc: model_dir(wc.study, wc.model),
        interpret_dir = lambda wc: os.path.join(model_dir(wc.study, wc.model), "interpret"),
        tax_flag      = _pp_tax_flag,
    log:
        OUTPUT_ROOT + "/{study}/models/{model}/logs/postprocess_interpret.log",
    threads: 20
    resources:
        partition   = "ei-gpu",
        runtime     = 48 * 60,
        mem_mb      = 128 * 1024,
        gpus        = 1,
        slurm_extra = "--gres=gpu:1",
    shell:
        r"""
        mkdir -p {params.interpret_dir}
        biomevae-interpret \
            --input {input.sgb_table} \
            --model-dir {params.model_dir} \
            --outdir {params.interpret_dir} \
            {params.tax_flag} \
            > {log} 2>&1
        """


rule postprocess_interpret_genus:
    input:
        model_pt  = OUTPUT_ROOT + "/{study}/models/{model}/model.pt",
        config    = OUTPUT_ROOT + "/{study}/models/{model}/config.json",
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        taxonomy  = lambda wc: data_path(wc.study, "phyla.tsv"),
    output:
        summary = OUTPUT_ROOT + "/{study}/models/{model}/interpret_genus/otu_latent_summary.tsv",
    params:
        model_dir           = lambda wc: model_dir(wc.study, wc.model),
        interpret_genus_dir = lambda wc: os.path.join(model_dir(wc.study, wc.model), "interpret_genus"),
    log:
        OUTPUT_ROOT + "/{study}/models/{model}/logs/postprocess_interpret_genus.log",
    threads: 20
    resources:
        partition   = "ei-gpu",
        runtime     = 48 * 60,
        mem_mb      = 128 * 1024,
        gpus        = 1,
        slurm_extra = "--gres=gpu:1",
    # Genus-level aggregation ALWAYS needs --taxonomy (independent of whether
    # the model itself is taxonomy-aware) because biomevae-interpret has to look
    # up each OTU's genus in the phyla.tsv to collapse the feature axis.
    # Passing --taxonomy for non-tax-aware models is safe: biomevae-interpret
    # only consumes it when --taxonomy-level is set (or when the model itself
    # requires it for rebuilding the graph).
    shell:
        r"""
        mkdir -p {params.interpret_genus_dir}
        biomevae-interpret \
            --input {input.sgb_table} \
            --model-dir {params.model_dir} \
            --outdir {params.interpret_genus_dir} \
            --taxonomy-level genus \
            --taxonomy {input.taxonomy} \
            > {log} 2>&1
        """
