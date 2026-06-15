# ──────────────────────────────────────────────────────────────────────
# aggregate.smk – Step 5: cross-model benchmarking / aggregation.
#
# The aggregation step runs a large sequence of biomevae CLIs:
#   * collect test_report.json into aggregate/test_summary.tsv
#   * plot training curves across all models
#   * biomevae-allcomp (vs NMF baseline)
#   * biomevae-benchmark-figure (+ recon-violin, pairwise-table,
#     hierarchy-figure, recon-scatter)
#   * biomevae-benchmark-figures-enterosignatures
#   * biomevae-benchmark-slides
#   * biomevae-interpret-compare
#
# All of that already lives in ``hpc/aggregate_results.sh``.  Rather
# than duplicate ~400 lines of glue, we reuse the existing script and
# replace its mamba wrapper with ``scripts/no_mamba_wrapper.sh`` so it
# can run inside Snakemake's already-activated environment.
# ──────────────────────────────────────────────────────────────────────

import os as _os

_AGG_SCRIPT_DEFAULT = _os.path.join(workflow.basedir, "..", "hpc", "aggregate_results.sh")
_NO_MAMBA = _os.path.join(workflow.basedir, "scripts", "no_mamba_wrapper.sh")

# When ``skip_interpret: true`` is set in config, the aggregate rule drops
# its dependency on ``<model>/interpret*`` outputs so the SHAP-based
# postprocess_interpret{,_genus} rules never fire. ``aggregate_results.sh``
# already no-ops its interpret-compare step when those directories are
# absent.
_SKIP_INTERPRET = bool(config.get("skip_interpret", False))


def _aggregate_interpret_inputs():
    if _SKIP_INTERPRET:
        return []
    return expand(
        OUTPUT_ROOT + "/{{study}}/models/{model}/interpret/otu_latent_summary.tsv",
        model=ALL_MODELS,
    )


def _aggregate_interpret_genus_inputs():
    if _SKIP_INTERPRET:
        return []
    return expand(
        OUTPUT_ROOT + "/{{study}}/models/{model}/interpret_genus/otu_latent_summary.tsv",
        model=ALL_MODELS,
    )


rule aggregate:
    input:
        test_reports = expand(
            OUTPUT_ROOT + "/{{study}}/models/{model}/test/test_report.json",
            model=ALL_MODELS,
        ),
        embed = expand(
            OUTPUT_ROOT + "/{{study}}/models/{model}/embed/embeddings.tsv",
            model=ALL_MODELS,
        ),
        recon = expand(
            OUTPUT_ROOT + "/{{study}}/models/{model}/embed/recon.tsv",
            model=ALL_MODELS,
        ),
        interpret = _aggregate_interpret_inputs(),
        interpret_genus = _aggregate_interpret_genus_inputs(),
        classify = expand(
            OUTPUT_ROOT + "/{{study}}/models/{model}/classify/classification_results.json",
            model=ALL_MODELS,
        ),
        sgb_table = lambda wc: data_path(wc.study, "sgb_table.tsv"),
        taxonomy  = lambda wc: data_path(wc.study, "phyla.tsv"),
    output:
        # ``hpc/aggregate_results.sh`` guarantees every file below is
        # written – either with real content (when ``biomevae-allcomp``
        # and the matching figure CLIs succeed) or as an empty stub
        # (``{}`` / header-only TSV / zero-byte PDF) when the optional
        # benchmarking sub-steps fail on the CPU-only ei-short
        # partition.  Snakemake treats them as hard outputs; any
        # missing file would otherwise re-break the DAG at the final
        # step.
        #
        # ``all_methods_vs_nmf.json`` and ``test_summary.tsv`` are the
        # data side-cars; the rest are the *aggregate figures* that
        # were previously produced as un-tracked side effects of
        # ``aggregate_results.sh``.  Declaring them here means the
        # ``rule all`` target only succeeds once every figure actually
        # exists on disk, so a silent benchmarking failure now stops
        # the workflow instead of leaving the aggregate directory
        # half-empty.
        allcomp           = OUTPUT_ROOT + "/{study}/models/aggregate/all_methods_vs_nmf.json",
        summary           = OUTPUT_ROOT + "/{study}/models/aggregate/test_summary.tsv",
        benchmark_metrics = OUTPUT_ROOT + "/{study}/models/aggregate/figures/benchmark_metrics.pdf",
        benchmark_ordi    = OUTPUT_ROOT + "/{study}/models/aggregate/figures/benchmark_ordinations.pdf",
        benchmark_violin  = OUTPUT_ROOT + "/{study}/models/aggregate/figures/benchmark_violin.pdf",
        hierarchy_metrics = OUTPUT_ROOT + "/{study}/models/aggregate/figures/hierarchy_metrics.pdf",
        recon_scatter     = OUTPUT_ROOT + "/{study}/models/aggregate/figures/recon_scatter.pdf",
        done              = touch(OUTPUT_ROOT + "/{study}/models/aggregate/.aggregate.done"),
    params:
        models_dir = lambda wc: models_dir(wc.study),
        agg_dir    = lambda wc: aggregate_dir(wc.study),
        agg_script = _AGG_SCRIPT_DEFAULT,
        no_mamba   = _NO_MAMBA,
        seeds      = EVAL_SEEDS_STR,
    log:
        OUTPUT_ROOT + "/{study}/models/aggregate/logs/aggregate.log",
    # HPC resources – mirror ``hpc/aggregate_results.slurm``:
    # ei-short partition, 2h wall-time, 4 CPUs, 16G RAM, no GPU.
    threads: 4
    resources:
        partition   = "ei-medium",
        runtime     = 26 * 60,
        mem_mb      = 16 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p {params.agg_dir}/logs
        # Reuse hpc/aggregate_results.sh but bypass its mamba wrapper so
        # every biomevae-* CLI call runs inside the already-activated env.
        MAMBA_EXEC={params.no_mamba} \
        CONDA_ENV=unused \
        MAMBA_ROOT_PREFIX=/tmp \
        EVAL_SEEDS="{params.seeds}" \
            bash {params.agg_script} \
                {params.models_dir} \
                {input.sgb_table} \
                {input.taxonomy} \
                > {log} 2>&1
        """
