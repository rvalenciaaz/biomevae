# ──────────────────────────────────────────────────────────────────────
# train.smk – Step 1: train every biomevae model variant on a study.
#
# One rule, one model.  Snakemake spreads the training jobs over cores
# (or SLURM nodes via a profile) using the ``{model}`` wildcard.  Each
# rule instance writes to
#   <output_root>/<study>/models/<model>/{model.pt, config.json,
#                                         training_log.tsv, …}
# which matches the layout expected by the downstream postprocess and
# classify rules.
# ──────────────────────────────────────────────────────────────────────


def _train_tax_flag(wildcards, input):
    """Only pass ``--taxonomy`` for taxonomy-aware models."""
    if MODELS[wildcards.model]["needs_tax"]:
        return f"--taxonomy {input.taxonomy}"
    return ""


def _train_metadata_flag(wildcards, input):
    """Pass ``--metadata`` for models that consume per-sample labels."""
    if MODELS[wildcards.model].get("needs_metadata", False):
        label_col = config.get("label", LABEL)
        return f"--metadata {input.metadata} --label-col {label_col}"
    return ""


def _train_metadata_input(wildcards):
    """Optional sample-metadata input; only required when the model uses it."""
    if MODELS[wildcards.model].get("needs_metadata", False):
        return data_path(wildcards.study, "sample_metadata.tsv")
    # Return an empty list so Snakemake does not attempt to require the file.
    return []


rule train_model:
    input:
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        taxonomy  = lambda wc: data_path(wc.study, "phyla.tsv"),
        metadata  = _train_metadata_input,
    output:
        model_pt = OUTPUT_ROOT + "/{study}/models/{model}/model.pt",
        config   = OUTPUT_ROOT + "/{study}/models/{model}/config.json",
    params:
        cmd      = lambda wc: MODELS[wc.model]["cmd"],
        outdir   = lambda wc: model_dir(wc.study, wc.model),
        tax_flag = _train_tax_flag,
        meta_flag = _train_metadata_flag,
        extra    = EXTRA_ARGS,
    log:
        OUTPUT_ROOT + "/{study}/models/{model}/logs/train.log",
    # HPC resources – mirror ``hpc/train_model.slurm``:
    # ei-gpu partition, 96h wall-time, 20 CPUs, 128G RAM, 1 GPU.
    threads: 20
    resources:
        partition   = "ei-gpu",
        runtime     = 120 * 60,        # 120 hours in minutes
        mem_mb      = 128 * 1024,
        gpus        = 1,
        slurm_extra = "--gres=gpu:1",
    shell:
        r"""
        mkdir -p {params.outdir}/logs
        {params.cmd} \
            --input {input.sgb_table} \
            --outdir {params.outdir} \
            {params.tax_flag} \
            {params.meta_flag} \
            {params.extra} \
            > {log} 2>&1
        """
