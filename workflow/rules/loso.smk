# ──────────────────────────────────────────────────────────────────────
# loso.smk – Leave-One-Study-Out evaluation rules.
#
# Pipeline shape (per disease group, e.g. CRC):
#
#     prepare_merged → train(model)                               (one job)
#                          ↓
#                       embed             (built into the trainer's outputs)
#                          ↓
#                       loso_classify_fold(model, held_out)        (per fold)
#                          ↓
#                       loso_aggregate(model)                        (per model)
#                          ↓
#                       loso_summary                                  (final)
#
# Diagnostics (control-anchor CORAL + MMD) are emitted alongside the
# aggregate so the "is DA needed?" metric travels with every model.
#
# This rule file is included from ``workflow/Snakefile.loso`` and shares
# helpers with ``rules/common.smk`` (model catalogue, ``study_out`` etc).
# ──────────────────────────────────────────────────────────────────────

import os as _os
from pathlib import Path as _Path


# ── config (LOSO-specific) ───────────────────────────────────────────
# All keys are documented in ``workflow/config/loso_crc.yaml``.
DISEASE_GROUP = config.get("disease_group", "crc")
LOSO_STUDIES = list(config["loso_studies"])
LOSO_MODELS = list(config.get("loso_models", [
    # ``hyp-philr-zinb`` was dropped from the catalogue because the
    # underlying PhILR-NB backbone has no separate ZINB likelihood (see
    # the comment at the top of ``hyperbolic_philrvae.py``); the entry
    # below now points at the real NB variant for the same role.
    "hyp-philrvae", "tree-dtm-vae",
    "diva-tree-dtm-vae", "diva-hyp-philr-nb", "diva-beta-vae",
    # Non-DIVA β-VAE: isolates the contribution of DIVA's domain-invariance
    # term over plain unsupervised representation learning on log1p counts.
    "beta-vae",
    # PhyloDIVA: DIVA + hierarchical clade critic (gradient-reversal
    # on internal-node abundances at multiple taxonomic depths) + BM
    # smoothness on the decoder + CORAL on z_x.  Built specifically to
    # rescue the count-likelihood backbones whose vanilla DIVA gain
    # collapses in LOSO (see results/loso_summary_after.tsv).
    "phylodiva-tree-dtm-vae", "phylodiva-hyp-philr-nb", "phylodiva-beta-vae",
    # TAXI variants on both Tree-DTM and Hyperbolic-PhILR backbones.
    "taxi-tree-dtm-vae", "taxi-hyp-philrvae",
    # Tree-based baselines that complete the picture without a learned
    # representation: XGBoost on raw features (the de-facto community
    # baseline) and XGBoost on per-study CORAL-aligned features (a
    # DIVA-spirit feature-level domain adaptation).
    "xgb-baseline", "xgb-coral",
    # CAPDA-VAE — the conditional-alignment + CLR-taxonomy model (this work).
    "capda-vae",
]))

# Where the merged multi-study dataset is written.
MERGED_ROOT = _os.path.join(OUTPUT_ROOT, f"_merged_{DISEASE_GROUP}")

# Which latent slice each model classifies on.  DIVA runs default to
# ``z_y`` (the class-anchored factor); non-DIVA runs use the full latent.
LOSO_LATENT_SLICE = dict(config.get("loso_latent_slice", {}))

# Per-model extra args appended to the global ``extra_args``.  Lets the
# LOSO config target Optuna search-space overrides at specific backbones
# (e.g. DIVA models receive ``--optuna-config configs/...diva.json``)
# without touching the others.  Mirrors the single-study extensibility.
LOSO_EXTRA_ARGS = dict(config.get("loso_extra_args", {}))
DIVA_OPTUNA_CONFIG = config.get("diva_optuna_config")


def _loso_model_extra(model: str) -> str:
    """Return the full ``extra_args`` string for a given model.

    Composed of:
      1. The global ``extra_args`` (already exposed as ``EXTRA_ARGS`` by
         ``common.smk``).
      2. ``loso_extra_args[<model>]`` if set in the config.
      3. ``--optuna-config <diva_optuna_config>`` if set globally and the
         model is a DIVA backbone (skipped when the per-model entry
         already supplies ``--optuna-config``).
    """
    parts = [EXTRA_ARGS]
    per_model = LOSO_EXTRA_ARGS.get(model, "")
    if per_model:
        parts.append(str(per_model))
    if (
        DIVA_OPTUNA_CONFIG
        and (
            model.startswith("diva-")
            or model.startswith("phylodiva-")
            or model.startswith("taxi-")
        )
        and "--optuna-config" not in " ".join(parts)
    ):
        parts.append(f"--optuna-config {DIVA_OPTUNA_CONFIG}")
    return " ".join(p for p in parts if p)


def _loso_classify_slice(model: str) -> str:
    """Default latent slice per model.

    DIVA / PhyloDIVA → ``z_y`` (the class-anchored factor).
    TAXI            → ``full`` excluding ``z_d``; the combined classifier head
                       reads ``[z_tau, z_rho]``, so the predictive embedding is
                       the concatenation of those two slices.  We default to
                       ``full`` because the per-factor slice files written for
                       TAXI store ``z_y == z_tau`` and ``z_x == z_rho`` — the
                       union of them is exactly ``full \\ z_d``.  Users who
                       want strict z_tau-only classification can override via
                       ``loso_latent_slice``.
    Others          → ``full``.
    """
    if model in LOSO_LATENT_SLICE:
        return LOSO_LATENT_SLICE[model]
    if model.startswith("diva-") or model.startswith("phylodiva-"):
        return "z_y"
    if model.startswith("taxi-"):
        return "full"
    return "full"


# ── model catalogue extension (DIVA backbones) ────────────────────────
# Mirror the schema of the per-study ``MODELS`` dict.  ``needs_metadata``
# is True for every DIVA variant because their training CLIs consume
# the merged ``sample_metadata.tsv`` to extract study and class labels.
LOSO_MODEL_SPECS = {
    "diva-tree-dtm-vae": {
        "cmd": "biomevae-train-diva-tree-dtm",
        "needs_tax": True,
        "needs_metadata": True,
    },
    "diva-hyp-philr-nb": {
        "cmd": "biomevae-train-diva-hyp-philrvae",
        "needs_tax": True,
        "needs_metadata": True,
    },
    # Non-taxonomy DIVA backbone — plain MLP encoder/decoder, MAE
    # reconstruction on log1p counts.  Isolates the contribution of
    # domain-invariance from any phylogenetic prior.
    "diva-beta-vae": {
        "cmd": "biomevae-train-diva-beta-vae",
        "needs_tax": False,
        "needs_metadata": True,
    },
    # PhyloDIVA backbones: same I/O contract as the DIVA wrappers, but
    # with the hierarchical clade critic + BM smoothness + CORAL on z_x.
    # All three need the taxonomy: the critic builds its aggregator
    # matrices from phyla.tsv, so even the β-VAE backbone (otherwise
    # tax-agnostic) takes ``--taxonomy``.
    "phylodiva-tree-dtm-vae": {
        "cmd": "biomevae-train-phylodiva-tree-dtm",
        "needs_tax": True,
        "needs_metadata": True,
    },
    "phylodiva-hyp-philr-nb": {
        "cmd": "biomevae-train-phylodiva-hyp-philrvae",
        "needs_tax": True,
        "needs_metadata": True,
    },
    "phylodiva-beta-vae": {
        "cmd": "biomevae-train-phylodiva-beta-vae",
        "needs_tax": False,
        "needs_metadata": True,
    },
    # TAXI: taxonomy-protected conditional invariance on top of the
    # Tree-DTM DIVA backbone.  ``z_tau`` (== DIVA ``z_y``) is never
    # scrubbed; only the residual ``z_rho`` (== DIVA ``z_x``) is passed
    # through the GRL'd conditional study critic.  Class-conditional
    # CORAL replaces the marginal CORAL of PhyloDIVA.  Designed to
    # rescue cohorts where the disease signal is itself taxonomic and
    # marginal study-invariance over-scrubs.
    "taxi-tree-dtm-vae": {
        "cmd": "biomevae-train-taxi-tree-dtm",
        "needs_tax": True,
        "needs_metadata": True,
    },
    # TAXI on the Hyperbolic-PhILR backbone — same conditional-invariance
    # mechanics as ``taxi-tree-dtm-vae`` but on the PhILR/Poincaré encoder.
    # Useful where the count likelihood is preferred (NB/multinomial via
    # PhILR) or where Euclidean over Poincaré matters for downstream tasks.
    "taxi-hyp-philrvae": {
        "cmd": "biomevae-train-taxi-hyp-philrvae",
        "needs_tax": True,
        "needs_metadata": True,
    },
    # XGBoost-on-raw-features baseline.  The "trainer" is a passthrough
    # featurisation that writes log1p(SGB) as embeddings.tsv; the actual
    # XGBoost classifier is fit per fold by biomevae-loso-classify, identical
    # to the VAE rows.  This is the row that says "is the VAE actually
    # learning anything useful at all, or could a tree on raw counts match it?".
    "xgb-baseline": {
        "cmd": "biomevae-train-xgb-baseline",
        "needs_tax": False,
        "needs_metadata": False,
    },
    # XGBoost with feature-level domain adaptation: per-study CORAL alignment
    # (Sun & Saenko 2016) before the same downstream XGBoost classifier.
    # Spiritual analog of DIVA's "factor out z_d, classify on what's left"
    # for tree-based models — the row that attributes any DIVA gain to
    # domain-invariance versus the latent-variable framework specifically.
    "xgb-coral": {
        "cmd": "biomevae-train-xgb-coral",
        "needs_tax": False,
        "needs_metadata": True,
    },
    # CAPDA-VAE: conditional-alignment + CLR-taxonomy VAE.  Its domain-aware
    # OOF invariant prediction is stacked with the raw log1p features as the
    # "embedding"; biomevae-loso-classify fits the same XGBoost as every other
    # row, so this is the VAE that finally matches the xgb-baseline while
    # resolving the taxonomy-vs-domain-invariance tension.  In the non-strict
    # pipeline the single full-dataset fit already produces leak-free per-study
    # OOF probabilities (each study predicted by a VAE that never saw it).
    "capda-vae": {
        "cmd": "biomevae-train-capda-vae",
        "needs_tax": True,
        "needs_metadata": True,
    },
}
# Merge into the global MODELS catalogue from ``common.smk`` so the
# generic ``train_model`` rule can dispatch the new commands.
for _name, _spec in LOSO_MODEL_SPECS.items():
    if _name not in MODELS:
        MODELS[_name] = _spec
ALL_MODELS = list(MODELS.keys())


# ── path helpers ─────────────────────────────────────────────────────
def loso_run_dir(model: str) -> str:
    """Where the unsupervised / DIVA training run for ``model`` lives."""
    return _os.path.join(OUTPUT_ROOT, "loso", DISEASE_GROUP, "models", model)


def loso_fold_dir(model: str, held_out: str) -> str:
    """Per-fold LOSO classification output dir."""
    return _os.path.join(
        OUTPUT_ROOT, "loso", DISEASE_GROUP,
        "folds", model, held_out,
    )


# ── prepare merged dataset (one-shot) ────────────────────────────────
rule loso_prepare:
    """Merge per-study extracts into a unified multi-study dataset."""
    output:
        sgb       = MERGED_ROOT + "/sgb_table.tsv",
        phyla     = MERGED_ROOT + "/phyla.tsv",
        metadata  = MERGED_ROOT + "/sample_metadata.tsv",
        manifest  = MERGED_ROOT + "/loso_manifest.json",
    params:
        data_root = DATA_ROOT,
        merged    = MERGED_ROOT,
        studies   = ",".join(LOSO_STUDIES),
    log:
        OUTPUT_ROOT + "/loso/" + DISEASE_GROUP + "/logs/prepare.log",
    threads: 2
    resources:
        partition   = "ei-medium",
        runtime     = 1500,
        mem_mb      = 16 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p $(dirname {log})
        biomevae-loso-prepare \
            --data-root {params.data_root} \
            --studies   {params.studies} \
            --outdir    {params.merged} \
            > {log} 2>&1
        """


# ── train (any model in MODELS) on the merged dataset ───────────────
def _loso_train_extra(model: str) -> str:
    """Per-model extra flags (e.g. label/study-col for DIVA / PhyloDIVA / TAXI)."""
    if (
        model.startswith("diva-")
        or model.startswith("phylodiva-")
        or model.startswith("taxi-")
        or model == "capda-vae"
    ):
        # DIVA / PhyloDIVA / TAXI / CAPDA CLIs accept --metadata + --label-col +
        # --study-col directly; the ``train_model`` rule below already
        # passes ``--metadata``, so we only need to add the label/study
        # columns.
        label_col = config.get("label", LABEL)
        return f"--label-col {label_col} --study-col study_name"
    return ""


rule loso_train_model:
    """Train one model variant on the *merged* multi-study dataset."""
    input:
        sgb_table = MERGED_ROOT + "/sgb_table.tsv",
        taxonomy  = MERGED_ROOT + "/phyla.tsv",
        metadata  = MERGED_ROOT + "/sample_metadata.tsv",
    output:
        model_pt    = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                      + "/models/{model}/model.pt",
        embeddings  = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                      + "/models/{model}/embeddings.tsv",
        config_json = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                      + "/models/{model}/config.json",
    params:
        cmd       = lambda wc: MODELS[wc.model]["cmd"],
        outdir    = lambda wc: loso_run_dir(wc.model),
        tax_flag  = lambda wc, input: (
            f"--taxonomy {input.taxonomy}"
            if MODELS[wc.model]["needs_tax"] else ""
        ),
        meta_flag = lambda wc, input: (
            f"--metadata {input.metadata}"
            if MODELS[wc.model].get("needs_metadata", False) else ""
        ),
        diva_extra = lambda wc: _loso_train_extra(wc.model),
        extra      = lambda wc: _loso_model_extra(wc.model),
    log:
        OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
        + "/models/{model}/logs/train.log",
    threads: 20
    resources:
        partition   = "ei-gpu",
        runtime     = 120 * 60,
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
            {params.diva_extra} \
            {params.extra} \
            > {log} 2>&1
        """


# ── per-fold classification on held-out study ───────────────────────
rule loso_classify_fold:
    """XGBoost on (train embeddings, train labels), eval on held-out."""
    input:
        embeddings = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                     + "/models/{model}/embeddings.tsv",
        metadata   = MERGED_ROOT + "/sample_metadata.tsv",
    output:
        results = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                  + "/folds/{model}/{held_out}/classification_results.json",
    params:
        outdir = lambda wc: loso_fold_dir(wc.model, wc.held_out),
        slice  = lambda wc: _loso_classify_slice(wc.model),
        label  = LABEL,
        seeds  = EVAL_SEEDS_STR,
    log:
        OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
        + "/folds/{model}/{held_out}/logs/classify.log",
    threads: 4
    resources:
        partition   = "ei-medium",
        runtime     = 1500,
        mem_mb      = 16 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p {params.outdir}/logs
        biomevae-loso-classify \
            --embeddings {input.embeddings} \
            --metadata   {input.metadata} \
            --label      {params.label} \
            --held-out-study {wildcards.held_out} \
            --latent-slice   {params.slice} \
            --outdir         {params.outdir} \
            --seeds          {params.seeds} \
            > {log} 2>&1
        """


# ── control-anchor diagnostic per model ──────────────────────────────
rule loso_diagnostic:
    """Pair-wise control-only CORAL + MMD on the trained latent."""
    input:
        embeddings = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                     + "/models/{model}/embeddings.tsv",
        metadata   = MERGED_ROOT + "/sample_metadata.tsv",
    output:
        summary = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                  + "/diagnostic/{model}/control_anchor_summary.json",
    params:
        outdir = lambda wc: _os.path.join(
            OUTPUT_ROOT, "loso", DISEASE_GROUP, "diagnostic", wc.model,
        ),
        slice  = lambda wc: _loso_classify_slice(wc.model),
        label  = LABEL,
        control_value = config.get("control_value", "healthy"),
    log:
        OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
        + "/diagnostic/{model}/logs/diagnostic.log",
    threads: 2
    resources:
        partition   = "ei-medium",
        runtime     = 1470,
        mem_mb      = 16 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p {params.outdir}/logs
        biomevae-loso-diagnostic \
            --embeddings {input.embeddings} \
            --metadata   {input.metadata} \
            --label      {params.label} \
            --control-value "{params.control_value}" \
            --latent-slice   {params.slice} \
            --outdir         {params.outdir} \
            > {log} 2>&1
        """


# ── per-model aggregation across folds ───────────────────────────────
rule loso_aggregate_model:
    """Concatenate every fold's classification result for one model."""
    input:
        folds = lambda wc: expand(
            OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
            + "/folds/" + wc.model + "/{held_out}/classification_results.json",
            held_out=LOSO_STUDIES,
        ),
        diag = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
               + "/diagnostic/{model}/control_anchor_summary.json",
    output:
        summary = OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
                  + "/models/{model}/loso_summary.tsv",
    log:
        OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
        + "/models/{model}/logs/loso_aggregate.log",
    threads: 1
    resources:
        partition   = "ei-medium",
        runtime     = 1455,
        mem_mb      = 4 * 1024,
        slurm_extra = "",
    run:
        import json as _json
        import pandas as _pd
        from pathlib import Path as _P

        _P(log[0]).parent.mkdir(parents=True, exist_ok=True)
        _P(output.summary).parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "model", "held_out_study",
            "balanced_accuracy", "f1_macro", "auroc",
            "balanced_accuracy_std", "f1_macro_std", "auroc_std",
            "n_train_samples", "n_eval_samples",
        ]

        rows = []
        for path, held in zip(input.folds, LOSO_STUDIES):
            try:
                with open(path) as fh:
                    data = _json.load(fh)
                xg = data["XGBoost"]
                rows.append({
                    "model": wildcards.model,
                    "held_out_study": held,
                    "balanced_accuracy": xg["balanced_accuracy"],
                    "f1_macro": xg["f1_macro"],
                    "auroc": xg.get("auroc"),
                    "balanced_accuracy_std": (
                        xg.get("across_seed_std", {}).get("balanced_accuracy")
                    ),
                    "f1_macro_std": (
                        xg.get("across_seed_std", {}).get("f1_macro")
                    ),
                    "auroc_std": (
                        xg.get("across_seed_std", {}).get("auroc")
                    ),
                    "n_train_samples": xg["n_train_samples"],
                    "n_eval_samples": xg["n_eval_samples"],
                })
            except Exception as exc:
                with open(log[0], "a") as logf:
                    logf.write(f"WARNING: {path}: {exc}\n")
        df = _pd.DataFrame(rows, columns=columns)
        df.to_csv(output.summary, sep="\t", index=False)


# ── meta summary across all (model, fold) combinations ──────────────
rule loso_summary:
    input:
        per_model = expand(
            OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
            + "/models/{model}/loso_summary.tsv",
            model=LOSO_MODELS,
        ),
        diagnostics = expand(
            OUTPUT_ROOT + "/loso/" + DISEASE_GROUP
            + "/diagnostic/{model}/control_anchor_summary.json",
            model=LOSO_MODELS,
        ),
    output:
        OUTPUT_ROOT + "/loso/" + DISEASE_GROUP + "/loso_summary.tsv",
    log:
        OUTPUT_ROOT + "/loso/" + DISEASE_GROUP + "/logs/loso_summary.log",
    threads: 1
    resources:
        partition   = "ei-medium",
        runtime     = 1455,
        mem_mb      = 4 * 1024,
        slurm_extra = "",
    run:
        import json as _json
        import pandas as _pd
        from pathlib import Path as _P

        _P(log[0]).parent.mkdir(parents=True, exist_ok=True)
        _P(output[0]).parent.mkdir(parents=True, exist_ok=True)

        frames = []
        with open(log[0], "a") as logf:
            for p in input.per_model:
                try:
                    df = _pd.read_csv(p, sep="\t")
                except _pd.errors.EmptyDataError:
                    logf.write(
                        f"WARNING: {p} is empty (no per-model rows); skipping.\n"
                    )
                    continue
                except Exception as exc:
                    logf.write(f"WARNING: {p}: {exc}; skipping.\n")
                    continue
                if df.empty:
                    logf.write(
                        f"WARNING: {p} has no rows; skipping.\n"
                    )
                    continue
                frames.append(df)
        if not frames:
            raise RuntimeError(
                "loso_summary: no per-model summaries had any rows. "
                f"See {log[0]} for per-file warnings."
            )
        merged = _pd.concat(frames, ignore_index=True, sort=False)

        # Attach per-model diagnostics (mean control-anchor MMD/CORAL).
        diag_lookup = {}
        for p in input.diagnostics:
            with open(p) as fh:
                data = _json.load(fh)
            # The folder name above the JSON file is the model key.
            model_name = _Path(p).parent.name
            diag_lookup[model_name] = {
                "ctrl_anchor_mmd_mean": data.get("mmd", {}).get("mean"),
                "ctrl_anchor_mmd_max":  data.get("mmd", {}).get("max"),
                "ctrl_anchor_coral_mean": data.get("coral", {}).get("mean"),
                "ctrl_anchor_coral_max":  data.get("coral", {}).get("max"),
                "n_studies_diag": data.get("n_studies_included"),
            }
        for col in (
            "ctrl_anchor_mmd_mean", "ctrl_anchor_mmd_max",
            "ctrl_anchor_coral_mean", "ctrl_anchor_coral_max",
            "n_studies_diag",
        ):
            merged[col] = merged["model"].map(
                lambda m: diag_lookup.get(m, {}).get(col)
            )
        merged.to_csv(output[0], sep="\t", index=False)
