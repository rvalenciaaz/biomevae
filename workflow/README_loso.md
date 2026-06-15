# Leave-One-Study-Out (LOSO) pipeline

Strict cross-cohort generalisation evaluation for biomevae.  Trains a
representation on N-1 studies, encodes the held-out study with the
trained encoder, trains an XGBoost classifier on the train-fold latent
embeddings, and evaluates it on the held-out study.  Repeats for every
study in the disease group.

## Why

The within-study 5-fold CV table at `results/meta_summary.tsv` is
optimistic: it does not test transfer across cohorts.  Cross-cohort
microbiome studies (Wirbel 2019, Thomas 2019) consistently report a
0.10–0.20 AUROC drop from within-study CV to held-out studies because
each cohort carries its own DNA-extraction / sequencing / population
fingerprint.  The LOSO table closes that gap by directly measuring the
quantity that matters for downstream deployment.

The CRC group (11 cMD studies) is the cleanest natural experiment in
the registry: same disease, large cross-cohort spread (within-study
AUROC ranges 0.56 – 0.95 in `results/meta_summary.tsv`), heterogeneous
geography and protocols.

## Pipeline shape

```
extract-microbiome-data per-study TSVs
            │
            ▼
   biomevae-loso-prepare        ← merge into one multi-study TSV
            │
            ▼
   biomevae-train-{model}       ← one job per model on the merged dataset
            │
            ▼
   biomevae-loso-classify       ← one job per (model, held-out study)
            │
            ▼
   biomevae-loso-diagnostic     ← control-anchor CORAL + MMD (per model)
            │
            ▼
   loso_summary.tsv           ← cross-model, cross-fold table
```

Snakemake rules live in `workflow/rules/loso.smk`; the entry point is
`workflow/Snakefile.loso` with config at `workflow/config/loso_crc.yaml`.

## Rigour parity with the single-study pipeline

The LOSO pipeline matches the single-study pipeline at every level
where stochastic noise can hide signal:

| Stage | Single-study | LOSO |
|---|---|---|
| Hyperparameter optimisation | Optuna, 100 trials, collapse-aware scoring, retrain-with-best | **same** — `extra_args: "--epochs 200 --optuna --optuna-trials 100"` is the LOSO default; every model (DIVA included) runs N trials and the best is retrained in the canonical outdir |
| Training-time seed variation | `seed = base + trial.number` per Optuna trial | **same** — built into `run_diva_optuna` (see `src/biomevae/cli/_diva_common.py`) |
| Evaluation-time seed replication | 5 seeds (`42 43 44 45 46`), per-seed XGBoost retrains pooled into mean ± std | **same** — `eval_seeds` config key, identical defaults; per-fold `classification_results.json` carries `across_seed_std` and `per_seed_metrics` |
| Best-trial artefacts | `optuna_best_params.json`, `optuna_trials.csv` | **same** — written into each model's `loso/<group>/models/<model>/` |

Hence the LOSO numbers in `loso_summary.tsv` are produced under the
same noise-control regime as `results/meta_summary.tsv`, and the two
tables are directly comparable.

To skip the Optuna search (e.g. for a smoke run or budget-constrained
HPC slot), drop the `--optuna` / `--optuna-trials` flags from
`extra_args` in `workflow/config/loso_crc.yaml`. Models will then train
once with their CLI defaults — fast but no HPO.

For per-model search-space overrides, see `loso_extra_args` and
`diva_optuna_config` in the config file. Default DIVA search space
ships at `configs/optuna_search_space_diva.template.json`.

## Quick start

Local machine:

```bash
snakemake -s workflow/Snakefile.loso \
          --configfile workflow/config/loso_crc.yaml \
          --cores 8
```

HPC (SLURM):

```bash
sbatch hpc/run_snakemake_loso.slurm
```

## Output layout

```
<output_root>/
├── _merged_crc/                       (one-shot, from biomevae-loso-prepare)
│   ├── sgb_table.tsv
│   ├── phyla.tsv
│   ├── sample_metadata.tsv            (with study_name column)
│   └── loso_manifest.json
└── loso/crc/
    ├── models/<model>/                (one trained encoder per model)
    │   ├── model.pt
    │   ├── embeddings.tsv             (full latent for any model)
    │   ├── embeddings_z_y.tsv         (DIVA: class-anchored slice)
    │   ├── embeddings_z_x.tsv         (DIVA: residual / domain-invariant slice)
    │   ├── embeddings_z_d.tsv         (DIVA: domain-anchored slice)
    │   ├── recon.tsv
    │   ├── training_log.tsv
    │   ├── config.json
    │   └── loso_summary.tsv           (per-fold AUROC/Bal-Acc/F1 for THIS model)
    ├── folds/<model>/<held_out_study>/
    │   ├── classification_results.json
    │   └── logs/classify.log
    ├── diagnostic/<model>/
    │   ├── control_anchor_coral.tsv
    │   ├── control_anchor_mmd.tsv
    │   └── control_anchor_summary.json
    └── loso_summary.tsv               ← top-level cross-model table
```

The top-level `loso_summary.tsv` columns:

| column | meaning |
|---|---|
| `model` | model key |
| `held_out_study` | study held out for this fold |
| `balanced_accuracy`, `f1_macro`, `auroc` | classifier metric on the held-out study, mean over the 5 evaluation seeds |
| `*_std` | unbiased across-seed standard deviation |
| `n_train_samples`, `n_eval_samples` | fold sizes |
| `ctrl_anchor_mmd_mean`, `ctrl_anchor_mmd_max` | control-only multi-bandwidth MMD² between studies, in the model's latent space |
| `ctrl_anchor_coral_mean`, `ctrl_anchor_coral_max` | Frobenius covariance distance between studies (CORAL diagnostic) |

## Models in the default sweep

Configured in `workflow/config/loso_crc.yaml` under `loso_models`:

| Model key | Inductive bias | What it tests |
|---|---|---|
| `hyp-philr-zinb` | hyperbolic latent + PhILR + ZINB likelihood | strongest non-DIVA compositional baseline |
| `tree-dtm-vae` | tree-softmax decoder + Dirichlet-Tree-Multinomial likelihood | strongest non-DIVA tree backbone |
| `diva-tree-dtm-vae` | tree + DTM + DIVA | target: do tree+DA combine? |
| `diva-hyp-philr-nb` | hyperbolic + PhILR + ZINB + DIVA | target: do all three combine? |
| `diva-beta-vae` | plain MLP + DIVA (no taxonomy, no tree) | **isolates DIVA's contribution** from any phylogenetic prior — `β-VAE` was 5/42 best in `meta_summary.tsv`, so a strong DIVA gain here means domain‑invariance, not phylogeny, is doing the work |
| `beta-vae` | plain MLP β-VAE (no DA, no taxonomy) | the missing comparison point for `diva-beta-vae`; with `loso_extra_args["beta-vae"]: "--log1p"` it shares the exact preprocessing of its DIVA counterpart, so the gap `diva-beta-vae − beta-vae` is a clean estimate of DIVA's contribution on plain MLP backbones |
| `xgb-baseline` | XGBoost on raw log1p(SGB) (no DA, no representation learning) | the de-facto community baseline (Wirbel 2019, Thomas 2019); trains XGBoost directly on the merged feature table per LOSO fold via the same StandardScaler + balanced class weights + 5-seed pipeline as every VAE row, so the numbers are directly comparable. If a VAE row does not beat this, the VAE is not earning its keep |
| `xgb-coral` | XGBoost on per-study CORAL-aligned features (feature-level DA, no representation learning) | the DIVA-spirit tree DA row: each study is whitened and re-coloured to a shared reference distribution before XGBoost sees it, so per-cohort mean and covariance fingerprints are removed without any class-label leakage. If a DIVA row beats this, the gain is attributable to the latent-variable factorisation specifically; if it doesn't, simpler feature-level alignment is enough |

## DIVA: what it does and why

DIVA (Ilse et al. 2020) factorises the VAE latent into three Gaussian
factors — `z_d` (domain-specific), `z_y` (class-specific), `z_x`
(residual) — each with its own conditional prior and an auxiliary
classifier:

* `p(z_d | d) = N(mu_d(d), sigma_d(d)^2)`  →  pushes batch / cohort
  / sequencing-platform variance into `z_d`.
* `p(z_y | y) = N(mu_y(y), sigma_y(y)^2)`  →  pushes disease variance
  into `z_y`.
* `p(z_x) = N(0, I)`                       →  whatever else is left.
* Auxiliary classifiers `q(d|z_d)` and `q(y|z_y)` enforce the
  factorisation by punishing latents that fail to predict their
  side-information.

For LOSO inference: we encode held-out samples with the shared encoder
(no domain or class label needed at inference) and classify on the
per-factor slice that should carry disease — `z_y` by default (override
per-model via `loso_latent_slice` in the config).

Three backbones are wrapped:

* `DIVATreeDTMVAE` (`models/diva_treedtmvae.py`) — DIVA on top of the
  Tree-DTM (Dirichlet-Tree Multinomial) compositional decoder.
* `DIVAHyperbolicPhILRNBVAE` (`models/diva_hyp_philrvae.py`) — DIVA on
  top of the hyperbolic PhILR-ZINB backbone.  Inference happens in
  tangent space (closed-form Gaussian KL); the joint tangent vector is
  mapped to the Poincaré ball with `expmap0` before decoding.
* `DIVABetaVAE` (`models/diva_betavae.py`) — DIVA on top of the plain
  MLP β-VAE.  No taxonomy, no tree, no compositional prior; MAE
  reconstruction on log1p counts.  Useful as a control for whether
  DIVA's gain is independent of phylogenetic structure.

The dependency-free DIVA core lives in `src/biomevae/models/diva.py`.

## Adapting to other disease groups

Copy `workflow/config/loso_crc.yaml` to `workflow/config/loso_<group>.yaml`,
edit `disease_group` and `loso_studies`, then:

```bash
sbatch hpc/run_snakemake_loso.slurm \
    --configfile workflow/config/loso_<group>.yaml
```

For multi-class groups (e.g. IBD with `UC` / `CD` / `nonIBD`) the
classifier reduces to a one-vs-rest macro-AUROC.  Override
`control_value` if the registry uses something other than `"healthy"`
for the reference class (the diagnostic step relies on it).

## Reading the table

* **No DA needed (gap < 0.05)** — non-DIVA backbones already generalise.
  Spend effort elsewhere (likelihood, phylogenetic prior, sample size).
* **Gap 0.05 – 0.15** — DIVA should close most of it.  If the
  `*_z_y` LOSO numbers are above the unadapted full-latent ones for the
  same backbone, the factorisation is doing its job.
* **Gap > 0.15 with DIVA still under-performing** — structural fix
  needed beyond DIVA (likelihood mismatch, batch confounded with
  outcome, …).  The pair-wise control-anchor MMD diagnostic flags
  which study pairs are responsible.

If `diva-beta-vae` improves over the plain `β-VAE` LOSO baseline by
roughly the same margin as `diva-tree-dtm-vae` over `tree-dtm-vae`, the
domain-invariance term — not the phylogenetic prior — is the active
ingredient.

### Reading the tree-based rows

`xgb-baseline` and `xgb-coral` use the same downstream classifier as
every VAE row; the only difference is the "embedding" they pass through:
`xgb-baseline` writes raw log1p(SGB) features; `xgb-coral` writes the
same features after per-study CORAL alignment.  Two consequences:

* The control-anchor diagnostic columns (`ctrl_anchor_coral_*`,
  `ctrl_anchor_mmd_*`) for these rows are computed in the *raw-feature*
  ambient space, not in a learned latent.  They are diagnostically
  meaningful (study fingerprint magnitude) but **not** numerically
  comparable to the latent-space diagnostics for the VAE rows — they
  live in a different ambient space with thousands of dimensions.
* For `xgb-coral`, `ctrl_anchor_coral_mean/max` should be near zero by
  construction (alignment did its job).  `ctrl_anchor_mmd_mean/max` is
  the more honest residual-drift number for that row, and a useful
  sanity check that the alignment is doing something.

References:
Ilse 2020 (DIVA), Sun & Saenko 2016 (CORAL), Gretton 2012 (MMD),
Wirbel 2019 *Nat Med* + Thomas 2019 *Nat Med* (CRC cross-cohort gap).
