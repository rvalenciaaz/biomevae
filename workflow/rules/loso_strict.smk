# ──────────────────────────────────────────────────────────────────────
# loso_strict.smk – Strict Leave-One-Study-Out evaluation rules.
#
# Difference vs. ``rules/loso.smk``:
#   * ``loso.smk`` trains a single encoder per model on the *full* merged
#     dataset; the held-out cohort therefore contributes to representation
#     learning and Optuna's hyperparameter selection (model-side leakage).
#   * ``loso_strict.smk`` trains one encoder per ``(model, held_out)``
#     fold on N-1 studies only.  Optuna's val pool excludes the held-out
#     cohort, the encoder weights never see it, and the held-out cohort
#     is embedded post-training via :command:`biomevae-loso-strict-encode`.
#
# Pipeline shape (per disease group, per fold):
#
#     prepare_merged → strict_fold(held_out)                         (per fold)
#                          ↓
#                      strict_train(model, held_out)                 (per fold × model)
#                          ↓
#                      strict_encode_holdout(model, held_out)        (per fold × model)
#                          ↓
#                      strict_classify(model, held_out)              (per fold × model)
#                          ↓
#                      strict_diagnostic(model, held_out)            (per fold × model)
#                          ↓
#                      strict_aggregate_model(model)                  (per model)
#                          ↓
#                      strict_summary                                  (final)
#
# Output root::
#
#     <output_root>/loso_strict/<disease_group>/
#         folds/<held_out>/{train,holdout}/{sgb_table,phyla,sample_metadata}.tsv
#         models/<model>/<held_out>/{model.pt,embeddings.tsv,config.json,
#                                    optuna_trials/,optuna_best_params.json,...}
#         models/<model>/<held_out>/holdout/embeddings*.tsv
#         classification/<model>/<held_out>/{classification_results.json,
#                                            embeddings_full.tsv}
#         diagnostic/<model>/<held_out>/control_anchor_summary.json
#         models/<model>/loso_summary.tsv
#         loso_summary.tsv
#
# This file is included from ``workflow/Snakefile.loso_strict`` and shares
# helpers with ``rules/common.smk`` / ``rules/loso.smk`` (model catalogue,
# OUTPUT_ROOT, EXTRA_ARGS, EVAL_SEEDS_STR).
# ──────────────────────────────────────────────────────────────────────

import os as _os
from pathlib import Path as _Path


# ── config (strict-LOSO-specific) ────────────────────────────────────
DISEASE_GROUP = config.get("disease_group", "crc")
LOSO_STUDIES = list(config["loso_studies"])
LOSO_MODELS = list(config.get("loso_models", [
    # ``hyp-philr-zinb`` is no longer a distinct entry in the catalogue
    # (PhILR-NB has no separate ZINB likelihood); use the NB variant.
    "hyp-philrvae", "tree-dtm-vae",
    "diva-tree-dtm-vae", "diva-hyp-philr-nb", "diva-beta-vae",
    "beta-vae",
    # PhyloDIVA — phylogeny-aware DA on top of each DIVA backbone.
    # Mirrors the loso.smk default so a strict-mode run uses the same
    # model catalogue.
    "phylodiva-tree-dtm-vae", "phylodiva-hyp-philr-nb", "phylodiva-beta-vae",
    # TAXI variants on both Tree-DTM and Hyperbolic-PhILR backbones.
    "taxi-tree-dtm-vae", "taxi-hyp-philrvae",
    "xgb-baseline", "xgb-coral",
    # CAPDA-VAE — the conditional-alignment + CLR-taxonomy model (this work).
    "capda-vae",
]))

# The full merged dataset is built once by ``rules/loso.smk:loso_prepare``;
# we reuse that rule via include.  The strict-mode root sits next to the
# non-strict ``loso/`` tree so both can coexist under the same output_root.
MERGED_ROOT = _os.path.join(OUTPUT_ROOT, f"_merged_{DISEASE_GROUP}")
STRICT_ROOT = _os.path.join(OUTPUT_ROOT, "loso_strict", DISEASE_GROUP)

LOSO_LATENT_SLICE = dict(config.get("loso_latent_slice", {}))
LOSO_EXTRA_ARGS = dict(config.get("loso_extra_args", {}))
DIVA_OPTUNA_CONFIG = config.get("diva_optuna_config")


def _strict_model_extra(model: str) -> str:
    """Compose ``extra_args`` for a model — same semantics as loso.smk."""
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


def _strict_classify_slice(model: str) -> str:
    if model in LOSO_LATENT_SLICE:
        return LOSO_LATENT_SLICE[model]
    if model.startswith("diva-") or model.startswith("phylodiva-"):
        return "z_y"
    if model.startswith("taxi-"):
        # TAXI's predictive head reads ``[z_tau, z_rho]`` (== z_y ∪ z_x);
        # ``full`` recovers that concatenation from the saved embeddings.
        return "full"
    return "full"


def _strict_train_extra(model: str) -> str:
    """Per-model extra flags (label/study col for DIVA / PhyloDIVA / TAXI / CAPDA)."""
    if (
        model.startswith("diva-")
        or model.startswith("phylodiva-")
        or model.startswith("taxi-")
        or model == "capda-vae"
    ):
        label_col = config.get("label", LABEL)
        return f"--label-col {label_col} --study-col study_name"
    return ""


# ── model catalogue extension (mirrors loso.smk) ─────────────────────
STRICT_MODEL_SPECS = {
    "diva-tree-dtm-vae": {
        "cmd": "biomevae-train-diva-tree-dtm",
        "needs_tax": True, "needs_metadata": True,
    },
    "diva-hyp-philr-nb": {
        "cmd": "biomevae-train-diva-hyp-philrvae",
        "needs_tax": True, "needs_metadata": True,
    },
    "diva-beta-vae": {
        "cmd": "biomevae-train-diva-beta-vae",
        "needs_tax": False, "needs_metadata": True,
    },
    "phylodiva-tree-dtm-vae": {
        "cmd": "biomevae-train-phylodiva-tree-dtm",
        "needs_tax": True, "needs_metadata": True,
    },
    "phylodiva-hyp-philr-nb": {
        "cmd": "biomevae-train-phylodiva-hyp-philrvae",
        "needs_tax": True, "needs_metadata": True,
    },
    "phylodiva-beta-vae": {
        "cmd": "biomevae-train-phylodiva-beta-vae",
        "needs_tax": False, "needs_metadata": True,
    },
    # TAXI: taxonomy-protected conditional invariance Tree-DTM DIVA variant.
    # See ``loso.smk`` for the long-form rationale.
    "taxi-tree-dtm-vae": {
        "cmd": "biomevae-train-taxi-tree-dtm",
        "needs_tax": True, "needs_metadata": True,
    },
    # TAXI on the Hyperbolic-PhILR backbone.
    "taxi-hyp-philrvae": {
        "cmd": "biomevae-train-taxi-hyp-philrvae",
        "needs_tax": True, "needs_metadata": True,
    },
    "xgb-baseline": {
        "cmd": "biomevae-train-xgb-baseline",
        "needs_tax": False, "needs_metadata": False,
    },
    "xgb-coral": {
        "cmd": "biomevae-train-xgb-coral",
        "needs_tax": False, "needs_metadata": True,
    },
    # CAPDA-VAE: conditional-alignment + CLR-taxonomy VAE whose domain-aware
    # OOF invariant prediction is stacked with the raw log1p features and
    # classified by the same XGBoost as every other model.  Needs the
    # taxonomy (multi-resolution features) and metadata (study + disease).
    "capda-vae": {
        "cmd": "biomevae-train-capda-vae",
        "needs_tax": True, "needs_metadata": True,
    },
}
for _name, _spec in STRICT_MODEL_SPECS.items():
    if _name not in MODELS:
        MODELS[_name] = _spec
ALL_MODELS = list(MODELS.keys())


# ── wildcard constraints ─────────────────────────────────────────────
# ``study`` / ``model`` are constrained globally in ``common.smk``.  The
# held-out study uses the same character class (study names are arbitrary
# tokens like ``ThomasAM_2019_c``).
wildcard_constraints:
    held_out = r"[A-Za-z0-9_\-\.]+",


# ── path helpers ─────────────────────────────────────────────────────
def strict_fold_dir(held_out: str) -> str:
    return _os.path.join(STRICT_ROOT, "folds", held_out)


def strict_train_dir(model: str, held_out: str) -> str:
    return _os.path.join(STRICT_ROOT, "models", model, held_out)


def strict_holdout_emb_dir(model: str, held_out: str) -> str:
    return _os.path.join(STRICT_ROOT, "models", model, held_out, "holdout")


def strict_classify_dir(model: str, held_out: str) -> str:
    return _os.path.join(STRICT_ROOT, "classification", model, held_out)


def strict_diagnostic_dir(model: str, held_out: str) -> str:
    return _os.path.join(STRICT_ROOT, "diagnostic", model, held_out)


# ── prepare merged dataset (one-shot, reuses loso_prepare's output) ──
# The full merged dataset is built once for both pipelines — strict and
# non-strict.  ``rules/loso.smk:loso_prepare`` already produces it; the
# strict pipeline simply depends on the same merged TSVs.
rule strict_prepare:
    """Merge per-study extracts into the full multi-study dataset."""
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
        STRICT_ROOT + "/logs/prepare.log",
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


# ── per-fold split: filter merged dataset to N-1 train + 1 holdout ──
rule strict_fold_split:
    """Slice the merged dataset into train / holdout bundles for one fold.

    Both bundles share feature ordering with the merged dataset, so the
    held-out sgb_table can be encoded by the train-fold model without
    column re-alignment downstream.
    """
    input:
        sgb       = MERGED_ROOT + "/sgb_table.tsv",
        phyla     = MERGED_ROOT + "/phyla.tsv",
        metadata  = MERGED_ROOT + "/sample_metadata.tsv",
    output:
        train_sgb  = STRICT_ROOT + "/folds/{held_out}/train/sgb_table.tsv",
        train_phy  = STRICT_ROOT + "/folds/{held_out}/train/phyla.tsv",
        train_meta = STRICT_ROOT + "/folds/{held_out}/train/sample_metadata.tsv",
        hold_sgb   = STRICT_ROOT + "/folds/{held_out}/holdout/sgb_table.tsv",
        hold_phy   = STRICT_ROOT + "/folds/{held_out}/holdout/phyla.tsv",
        hold_meta  = STRICT_ROOT + "/folds/{held_out}/holdout/sample_metadata.tsv",
        manifest   = STRICT_ROOT + "/folds/{held_out}/fold_manifest.json",
    params:
        outdir = lambda wc: strict_fold_dir(wc.held_out),
        merged = MERGED_ROOT,
    log:
        STRICT_ROOT + "/folds/{held_out}/logs/strict_fold.log",
    threads: 1
    resources:
        partition   = "ei-medium",
        runtime     = 1470,
        mem_mb      = 8 * 1024,
        slurm_extra = "",
    shell:
        r"""
        mkdir -p $(dirname {log})
        biomevae-loso-strict-fold \
            --merged   {params.merged} \
            --held-out {wildcards.held_out} \
            --outdir   {params.outdir} \
            > {log} 2>&1
        """


# ── train one model on one fold's N-1 studies ───────────────────────
rule strict_train_fold:
    """Train one model variant on the train-fold (N-1 studies) only.

    Optuna runs entirely inside this job; every trial sees the train-
    fold dataset and never the held-out cohort.  Output layout matches
    the non-strict ``loso_train_model`` rule, but the artefact directory
    is parameterised on ``{held_out}`` so the runs are independent.
    """
    input:
        sgb_table = STRICT_ROOT + "/folds/{held_out}/train/sgb_table.tsv",
        taxonomy  = STRICT_ROOT + "/folds/{held_out}/train/phyla.tsv",
        metadata  = STRICT_ROOT + "/folds/{held_out}/train/sample_metadata.tsv",
    output:
        model_pt    = STRICT_ROOT + "/models/{model}/{held_out}/model.pt",
        embeddings  = STRICT_ROOT + "/models/{model}/{held_out}/embeddings.tsv",
        config_json = STRICT_ROOT + "/models/{model}/{held_out}/config.json",
    params:
        cmd       = lambda wc: MODELS[wc.model]["cmd"],
        outdir    = lambda wc: strict_train_dir(wc.model, wc.held_out),
        tax_flag  = lambda wc, input: (
            f"--taxonomy {input.taxonomy}"
            if MODELS[wc.model]["needs_tax"] else ""
        ),
        meta_flag = lambda wc, input: (
            f"--metadata {input.metadata}"
            if MODELS[wc.model].get("needs_metadata", False) else ""
        ),
        diva_extra = lambda wc: _strict_train_extra(wc.model),
        extra      = lambda wc: _strict_model_extra(wc.model),
    log:
        STRICT_ROOT + "/models/{model}/{held_out}/logs/train.log",
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


# ── encode the held-out cohort with the strict-trained model ────────
rule strict_encode_holdout:
    """Apply the trained encoder to the held-out study's samples.

    Writes ``embeddings.tsv`` (and DIVA per-factor slices when
    applicable) under ``models/<model>/<held_out>/holdout/``.  For
    ``xgb-coral`` the held-out features are CORAL-aligned to a
    train-only reference so the alignment never sees the held-out
    cohort's stats.
    """
    input:
        model_pt    = STRICT_ROOT + "/models/{model}/{held_out}/model.pt",
        config_json = STRICT_ROOT + "/models/{model}/{held_out}/config.json",
        hold_sgb    = STRICT_ROOT + "/folds/{held_out}/holdout/sgb_table.tsv",
        hold_phy    = STRICT_ROOT + "/folds/{held_out}/holdout/phyla.tsv",
        hold_meta   = STRICT_ROOT + "/folds/{held_out}/holdout/sample_metadata.tsv",
        train_sgb   = STRICT_ROOT + "/folds/{held_out}/train/sgb_table.tsv",
        train_meta  = STRICT_ROOT + "/folds/{held_out}/train/sample_metadata.tsv",
    output:
        embeddings = STRICT_ROOT + "/models/{model}/{held_out}/holdout/embeddings.tsv",
    params:
        model_dir  = lambda wc: strict_train_dir(wc.model, wc.held_out),
        outdir     = lambda wc: strict_holdout_emb_dir(wc.model, wc.held_out),
        tax_flag   = lambda wc, input: (
            f"--taxonomy {input.hold_phy}"
            if MODELS[wc.model]["needs_tax"] else ""
        ),
        meta_flag  = lambda wc, input: f"--metadata {input.hold_meta}",
        coral_flag = lambda wc, input: (
            f"--reference-input {input.train_sgb} "
            f"--reference-metadata {input.train_meta}"
            if wc.model == "xgb-coral" else ""
        ),
    log:
        STRICT_ROOT + "/models/{model}/{held_out}/logs/encode_holdout.log",
    threads: 4
    resources:
        partition   = "ei-gpu",
        runtime     = 1500,
        mem_mb      = 32 * 1024,
        gpus        = 1,
        slurm_extra = "--gres=gpu:1",
    shell:
        r"""
        mkdir -p {params.outdir}
        biomevae-loso-strict-encode \
            --model-dir {params.model_dir} \
            --input     {input.hold_sgb} \
            {params.meta_flag} \
            {params.tax_flag} \
            {params.coral_flag} \
            --outdir    {params.outdir} \
            > {log} 2>&1
        """


# ── concat train + holdout embeddings into one file for the classifier ─
rule strict_concat_embeddings:
    """Concatenate train-fold + held-out embeddings (and per-factor slices).

    ``biomevae-loso-classify`` consumes a single embeddings TSV indexed by
    sample ID and splits internally on the study column, so we provide
    the union here.  For DIVA models we also concatenate the
    ``embeddings_z_y.tsv`` / ``embeddings_z_x.tsv`` / ``embeddings_z_d.tsv``
    slices so the slice-aware classification path works.
    """
    input:
        train_emb = STRICT_ROOT + "/models/{model}/{held_out}/embeddings.tsv",
        hold_emb  = STRICT_ROOT + "/models/{model}/{held_out}/holdout/embeddings.tsv",
    output:
        full_emb = STRICT_ROOT + "/classification/{model}/{held_out}/embeddings.tsv",
    log:
        STRICT_ROOT + "/classification/{model}/{held_out}/logs/concat.log",
    threads: 1
    resources:
        partition   = "ei-medium",
        runtime     = 1455,
        mem_mb      = 8 * 1024,
        slurm_extra = "",
    run:
        import pandas as _pd
        from pathlib import Path as _P

        out_dir = _P(output.full_emb).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        train_dir = _P(input.train_emb).parent
        hold_dir = _P(input.hold_emb).parent

        for fname in (
            "embeddings.tsv",
            "embeddings_z_d.tsv", "embeddings_z_y.tsv", "embeddings_z_x.tsv",
        ):
            tp = train_dir / fname
            hp = hold_dir / fname
            if not tp.exists() or not hp.exists():
                # Per-factor slices only exist for DIVA models — skip
                # silently for others.  ``embeddings.tsv`` always exists.
                if fname != "embeddings.tsv":
                    continue
                with open(log[0], "a") as logf:
                    logf.write(
                        f"WARNING: missing {tp} or {hp}; skipping {fname}\n"
                    )
                continue
            df_train = _pd.read_csv(tp, sep="\t", index_col=0)
            df_hold = _pd.read_csv(hp, sep="\t", index_col=0)
            df = _pd.concat([df_train, df_hold], axis=0)
            df.to_csv(out_dir / fname, sep="\t")


# ── per-fold classification on the held-out study ───────────────────
rule strict_classify_fold:
    """XGBoost on (train embeddings, train labels), eval on held-out."""
    input:
        embeddings = STRICT_ROOT + "/classification/{model}/{held_out}/embeddings.tsv",
        metadata   = MERGED_ROOT + "/sample_metadata.tsv",
    output:
        results = STRICT_ROOT + "/classification/{model}/{held_out}/classification_results.json",
    params:
        outdir = lambda wc: strict_classify_dir(wc.model, wc.held_out),
        slice  = lambda wc: _strict_classify_slice(wc.model),
        label  = LABEL,
        seeds  = EVAL_SEEDS_STR,
    log:
        STRICT_ROOT + "/classification/{model}/{held_out}/logs/classify.log",
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


# ── per-fold control-anchor diagnostic on the strict-trained latent ─
rule strict_diagnostic_fold:
    """Pair-wise control-only CORAL + MMD on the strict-trained latent.

    Unlike the non-strict pipeline (which runs one diagnostic per model
    on the full-encoded latent), each fold has its own trained encoder,
    so the diagnostic must run per ``(model, held_out)`` and be averaged
    across folds.
    """
    input:
        embeddings = STRICT_ROOT + "/classification/{model}/{held_out}/embeddings.tsv",
        metadata   = MERGED_ROOT + "/sample_metadata.tsv",
    output:
        summary = STRICT_ROOT + "/diagnostic/{model}/{held_out}/control_anchor_summary.json",
    params:
        outdir = lambda wc: strict_diagnostic_dir(wc.model, wc.held_out),
        slice  = lambda wc: _strict_classify_slice(wc.model),
        label  = LABEL,
        control_value = config.get("control_value", "healthy"),
    log:
        STRICT_ROOT + "/diagnostic/{model}/{held_out}/logs/diagnostic.log",
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
rule strict_aggregate_model:
    """Concatenate every fold's classification + diagnostic for one model."""
    input:
        folds = lambda wc: expand(
            STRICT_ROOT + "/classification/" + wc.model
            + "/{held_out}/classification_results.json",
            held_out=LOSO_STUDIES,
        ),
        diags = lambda wc: expand(
            STRICT_ROOT + "/diagnostic/" + wc.model
            + "/{held_out}/control_anchor_summary.json",
            held_out=LOSO_STUDIES,
        ),
    output:
        summary = STRICT_ROOT + "/models/{model}/loso_summary.tsv",
    log:
        STRICT_ROOT + "/models/{model}/logs/loso_aggregate.log",
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
            "ctrl_anchor_mmd_mean", "ctrl_anchor_mmd_max",
            "ctrl_anchor_coral_mean", "ctrl_anchor_coral_max",
        ]

        rows = []
        for path, diag_path, held in zip(input.folds, input.diags, LOSO_STUDIES):
            try:
                with open(path) as fh:
                    data = _json.load(fh)
                xg = data["XGBoost"]
                row = {
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
                }
                with open(diag_path) as fh:
                    diag = _json.load(fh)
                row.update({
                    "ctrl_anchor_mmd_mean": diag.get("mmd", {}).get("mean"),
                    "ctrl_anchor_mmd_max":  diag.get("mmd", {}).get("max"),
                    "ctrl_anchor_coral_mean": diag.get("coral", {}).get("mean"),
                    "ctrl_anchor_coral_max":  diag.get("coral", {}).get("max"),
                })
                rows.append(row)
            except Exception as exc:
                with open(log[0], "a") as logf:
                    logf.write(f"WARNING: {path}: {exc}\n")
        df = _pd.DataFrame(rows, columns=columns)
        df.to_csv(output.summary, sep="\t", index=False)


# ── final cross-model summary ────────────────────────────────────────
rule strict_summary:
    input:
        per_model = expand(
            STRICT_ROOT + "/models/{model}/loso_summary.tsv",
            model=LOSO_MODELS,
        ),
    output:
        STRICT_ROOT + "/loso_summary.tsv",
    log:
        STRICT_ROOT + "/logs/loso_summary.log",
    threads: 1
    resources:
        partition   = "ei-medium",
        runtime     = 1455,
        mem_mb      = 4 * 1024,
        slurm_extra = "",
    run:
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
                "strict_summary: no per-model summaries had any rows. "
                f"See {log[0]} for per-file warnings."
            )
        merged = _pd.concat(frames, ignore_index=True, sort=False)
        merged.to_csv(output[0], sep="\t", index=False)
