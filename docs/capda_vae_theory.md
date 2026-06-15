# CAPDA-VAE: cross-cohort and single-study variants

CAPDA-VAE (**C**lass-conditional, **A**dversary-free, **P**hylogenetic
**D**omain-**A**ligned VAE) is implemented in
[`src/biomevae/models/capda_vae.py`](../src/biomevae/models/capda_vae.py).
It comes in two flavours that share one network and one training loop:

| Variant | Entry point | Pipeline | Module function |
|---|---|---|---|
| Cross-cohort (LOSO) | `biomevae-train-capda-vae` | `Snakefile.loso{,_strict}` | `capda_fit` |
| Single-study | `biomevae-train-capda-vae-ss` | `Snakefile` / `Snakefile.meta` | `capda_fit_single_study` |

This note explains *why* a single-study variant is well-defined and what it
keeps from the cross-cohort theory in
[`taxonomy_vs_domain_invariance_theorem.md`](taxonomy_vs_domain_invariance_theorem.md).

## 1. What CAPDA does cross-cohort

The cross-cohort CAPDA resolves the proven tension between the *taxonomy*
inductive bias and *domain invariance* (the boxed inequalities **(T)**, **(I)**,
**(C)** of the theorem). It has three ingredients:

1. **Multi-resolution taxonomy bias.** The encoder sees per-species CLR
   coordinates *plus* abundances aggregated up the taxonomy
   (genus / family / order / phylum) — `build_vae_input`.
2. **Adversary-free *conditional* domain invariance.** Per-`(study, class)`
   latent means and covariances are pulled toward a shared per-class reference
   (`_conditional_alignment`, `_conditional_cov_alignment`). Aligning *within a
   class* removes the study nuisance from `P(z | y)` without erasing the
   between-class signal that *marginal* DA-VAEs destroy under label shift.
3. **Leak-free, domain-aware OOF stacking.** Class-head probabilities are
   produced out-of-fold by training on the *other* studies and predicting the
   held-out one (mirroring LOSO), then stacked with the `log1p` species
   features for the downstream XGBoost.

## 2. Why a single-study variant is well-defined

In the single-study / meta pipeline (`Snakefile`, `Snakefile.meta`) every
training job runs on **one cohort**. There is no second study to be invariant
*to*, so ingredient (2) has nothing to align — and indeed
`_conditional_alignment` returns exactly zero whenever a class is observed in
fewer than two domains (`present.numel() < 2`). The conditional-invariance term
is therefore **not removed, it is dormant**: it re-activates automatically if a
within-study sub-cohort label (e.g. a sequencing batch) is supplied via
`--study-col` with ≥ 2 levels.

That leaves the two ingredients that transfer unchanged to a single cohort:

* **(1) the multi-resolution taxonomy bias** — identical `build_vae_input`; and
* **(3) leak-free stacking** — but the *domain-aware* OOF of LOSO (train on
  other studies) has no analogue with one cohort, so it is replaced by its
  natural within-study counterpart: a **stratified K-fold** OOF
  (`capda_fit_single_study`). Each sample is scored by a VAE trained on folds
  that exclude it, so the class-probability columns handed to the downstream
  classifier are not in-sample-leaky. The final VAE (fit on all labelled
  samples) is saved for the embed / encode step.

The single-study CAPDA is thus the same supervised, taxonomy-biased VAE with
honest OOF stacking, degrading gracefully to the regime where domain invariance
is vacuous — exactly the regime a single cohort lives in.

## 3. Pipeline integration

`capda-vae` is a first-class entry in the single-study model catalogue
(`workflow/rules/common.smk`) and is **not** `cross_study_only`, so — unlike the
`diva-*` / `phylodiva-*` / `taxi-*` wrappers — it survives the single-study
filter. It satisfies the full per-model contract the standard / meta pipeline
relies on:

| Step | Rule | Artifact |
|---|---|---|
| train | `train_model` | `model.pt`, `config.json`, `oof_embeddings.tsv` |
| test | `postprocess_test` | `test/test_report.json`, `test/embeddings.tsv` |
| embed | `postprocess_embed` | `embed/embeddings.tsv` *(leak-free OOF passthrough)*, `embed/recon.tsv` |
| interpret | `postprocess_interpret{,_genus}` | `interpret*/otu_latent_summary.tsv` (SHAP via `CAPDARawEncoderMean`) |
| classify | `classify` | `classify/classification_results.json` |

The key subtlety is that **`biomevae-embed` passes the stored leak-free OOF
table straight through** to `biomevae-classify` (re-deriving the class
probabilities in-sample would leak the label); fresh / held-out samples not
present in the OOF table fall back to the final VAE's probabilities, exactly as
the LOSO encode step does (`capda_encode`).
