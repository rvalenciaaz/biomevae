# biomevae

A family of variational autoencoders for sparse [samples × features] microbiome
abundance tables, including compositional, taxonomy-aware, hyperbolic, and
phylogeny-aware domain-adaptive (DIVA / PhyloDIVA) variants.

The package learns latent representations of microbiome feature tables —
typically **S**pecies-level **G**enome **B**ins (SGBs) — and extends the base
VAE with taxonomy-aware decoders, hyperbolic latents, tree-structured
likelihoods (Dirichlet-tree-multinomial), PhILR (isometric log-ratio)
compositional coordinates, and DIVA-style latent partitioning with adversarial
domain adaptation across studies.

This README focuses on the **HPC pipelines** (single-study automation and
the general-purpose 3-stage SLURM submission scripts) plus the
cross-cutting **reproducibility contract** that every CLI in the
repository participates in. The rest of the documentation has been
split into topic-specific READMEs so that this page stays focused on
running batch jobs on a cluster.

## Documentation map

| Page | Contents |
|------|----------|
| [`README_installation.md`](README_installation.md) | `pip` / `mamba` installation, optional extras, pinned reference environment, input TSV format, taxonomy file layout |
| [`README_training.md`](README_training.md) | Every training CLI (β-VAE, vanilla, taxonomy-aware, graph, TreeDTM-VAE, HG-VAE-ZI, tree-prior, PhILR-VAE, hyperbolic PhILR-VAE, FlowXFormer, phylogenetic fusion, hyperbolic, hyperbolic+tax, DIVA / PhyloDIVA variants on β-VAE, hyperbolic PhILR and TreeDTM backbones), multi-model training-curve plotting, Optuna hyperparameter search |
| [`README_evaluation.md`](README_evaluation.md) | Embedding extraction, SHAP/MI/Spearman interpretation, held-out reconstruction tests, NMF baseline + Gabriel bi-cross-validation, single/multi method comparison vs. NMF, metadata classification from embeddings |
| [`README_figures.md`](README_figures.md) | Benchmark bar charts, enterosignature figures, Beamer slide deck, reconstruction scatter plots, violin distributions, per-taxonomy-level hierarchy breakdowns, pairwise significance heatmaps, cross-model interpretation comparison, UMAP ordinations, posterior-collapse diagnostics, sparsity-aware metrics, mathematical theory docs |
| [`workflow/README.md`](workflow/README.md) | Snakemake workflow (recommended): single-study and meta pipelines, per-rule SLURM resources, profile setup, single-target re-runs |
| [`docs/`](docs/) | Markdown + LaTeX theory write-ups for every model architecture |

If you are just getting started, read
[`README_installation.md`](README_installation.md) first, then follow
[`README_training.md`](README_training.md) to train a single model,
[`README_evaluation.md`](README_evaluation.md) to evaluate it, and
[`README_figures.md`](README_figures.md) to generate publication-ready
figures. Once the single-model workflow is comfortable, return to this
page to launch the full HPC pipeline.

## Reproducibility and seeds

Every post-training evaluation in `biomevae` — classification, VAE
reconstruction, NMF baselines, enterosignature agreement, pairwise
significance tables, figure generation — is repeated across the five
canonical seeds

```python
DEFAULT_EVAL_SEEDS = (42, 43, 44, 45, 46)
```

exported from `biomevae.classify.DEFAULT_EVAL_SEEDS` and consumed by
`biomevae.classify.evaluate_classifiers`,
`biomevae.reconstruction.cross_validate_nmf_multi_seed`,
`biomevae.reconstruction.cross_validate_vae_multi_seed`,
`biomevae.reconstruction.compare_all_methods_multi_seed`, and the
corresponding CLIs (`biomevae-classify`, `biomevae-nmf`,
`biomevae-comparetonmf`, `biomevae-allcomp`, `biomevae-pairwise-table`,
`biomevae-benchmark-figure`, ...).

### Aggregation contract

For each seed every evaluator runs the full `n_splits * n_repeats`
cross-validation protocol, records the mean metrics as a per-seed
summary, and then the multi-seed wrapper pools results as follows:

- **`per_seed_metrics` / `per_seed_mean_metrics`** — the full per-seed
  fold summary, keyed by seed string.
- **`fold_metrics`** — concatenation of every seed’s folds (used for
  paired tests that need matched fold partitions).
- **`mean_metrics`** — **mean of per-seed means**, not the naive pooled
  fold mean. This matches the Bouthillier et al. (2021) run-to-run
  variance framing so that correlated folds within the same seed do
  not inflate the effective sample size.
- **`std_metrics` / `across_seed_std`** — **unbiased standard deviation
  of the per-seed means** (`ddof=1`). Evaluators that trained with a
  single seed return `None` here.
- **`metadata['aggregation']`** — set to `"mean_of_per_seed_means"` so
  downstream consumers can detect the semantics unambiguously.
- **`metadata['pooled_fold_mean_metrics']` /
  `pooled_fold_std_metrics`** — the legacy naive pooled view is
  retained under dedicated keys for backwards compatibility.
- **`metadata['seeds']` / `metadata['n_seeds']`** — the resolved seed
  list and count.

The legacy `--seed N` / `seed=N` kwarg still works for single-seed
reruns; passing an explicit `seeds=[...]` list overrides the default.

### Deterministic training

`biomevae.utils.set_global_seed` is the single entry point for seeding a
process. It seeds Python's `random`, NumPy's global and default
generator, `torch.manual_seed` / `torch.cuda.manual_seed_all`, exports
`PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and sets
`torch.backends.cudnn.deterministic=True` / `benchmark=False`.
Setting the environment variable

```bash
export BIOMEVAE_DETERMINISTIC_TORCH=1
```

additionally turns on `torch.use_deterministic_algorithms(True,
warn_only=True)` so unsupported kernels raise rather than silently
produce non-deterministic output. `set_global_seed` is called at the
top of every per-seed training and evaluation pass; the training loops
also snapshot and restore the caller's RNG state at function
boundaries, so repeated calls with the same seed produce bitwise
identical checkpoints and the caller's global state is unaffected.

### Classifier determinism

The XGBoost baseline inside `biomevae-classify` now takes
`random_state=seed` and `n_jobs=1` (the parallel histogram reduction
is non-deterministic with `n_jobs=-1`). Logistic Regression, Random
Forest, SVM, and Gradient Boosting all forward the resolved per-seed
integer into their `random_state` constructors, so classification
results are reproducible across machines that share the pinned
environment.

### Paired statistical tests

`biomevae-pairwise-table` defaults to `--test seed`, which reads the
per-seed mean metrics from `metadata['per_seed_mean_metrics']` and
reports three p-values for each method pair in the TSV output:

1. `p_value_sign` — two-sided paired sign test.
2. `p_value_wilcoxon` — paired Wilcoxon signed-rank test.
3. `p_value_tcorrected` — **Nadeau & Bengio (2003) corrected paired
   t-test** with variance inflation `(1/n + ρ/(1−ρ))` where
   `ρ = 1 − train_fraction`. Controlled via `--train-fraction`
   (default `0.9`, matching the cross-validation CLIs).

The canonical `p_value` column is the Nadeau–Bengio corrected t value
and is the one forwarded to the Benjamini–Hochberg / Bonferroni
adjustments and rendered in the LaTeX tables and PDF heatmaps. Pass
`--test fold` to fall back to the legacy sign test on pooled fold
metrics (only meaningful when every method was evaluated on identical
fold partitions).

### Provenance block

Every result JSON — `classification_results.json`,
`nmf_summary.json`, `all_methods_vs_nmf.json`, the per-model test
reports, and the intermediate files produced by
`cross_validate_nmf_multi_seed` / `cross_validate_vae_multi_seed` —
now embeds a provenance block under `metadata['provenance']` that
records:

- `captured_at` — UTC ISO timestamp.
- `git_sha` + `git_dirty` — commit hash and working-tree cleanliness.
- `platform.{system, release, machine, python_version}`.
- `packages` — installed versions of `biomevae`, `numpy`, `pandas`,
  `scipy`, `scikit-learn`, `torch`, `xgboost` (and any extras you
  pass to `capture_provenance(packages=...)`).
- `torch` — version, CUDA availability, CUDA/cuDNN versions,
  deterministic-algorithm flags.
- `threads` — OMP/MKL/OpenBLAS/CUBLAS workspace settings and
  `torch.get_num_threads()`.
- `seeds` — the exact seed list pooled for that result.

Figures emitted by `biomevae-benchmark-figure`,
`biomevae-benchmark-figures-enterosignatures`, and the slide deck CLI
propagate the same block into their input JSON, so every figure in the
paper is traceable back to a known-good environment.

### Pinned reference environment

`environment.lock.yml` (conda/mamba) and `requirements.lock.txt` (pip)
capture the exact package versions used to produce every number in
the biomevae reports. Install them when reproducing results from the
provenance block:

```bash
mamba env create -n biomevae -f environment.lock.yml
mamba activate biomevae
pip install -e .
```

or, without conda:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e .
```

The pinned versions are `numpy==1.26.4`, `pandas==2.2.2`,
`scipy==1.13.0`, `scikit-learn==1.4.2`, `torch==2.2.2`,
`xgboost==2.0.3`, `statsmodels==0.14.2`, `matplotlib==3.8.4`,
`snakemake==8.10.7`, `optuna==3.6.1`, `umap-learn==0.5.6`,
`geoopt==0.5.0`, and `shap==0.45.0`. See
[`README_installation.md`](README_installation.md) for the full
installation matrix.

## Data extraction

Data extraction is handled by a separate package,
[extract-microbiome-data](https://github.com/rvazdev-ex/extract-microbiome-data),
which provides extractors for curatedMetagenomicData, the Human Microbiome
Project, and MGnify/EBI. Every extractor produces the three biomevae-compatible
files (`sgb_table.tsv`, `phyla.tsv`, `sample_metadata.tsv`) that the commands
below expect.

Install it alongside biomevae on a machine with internet access and follow
the usage examples in its README. A quick example for a single
curatedMetagenomicData study (e.g. `LiJ_2017` — CRC, ~110 samples):

```bash
python -m curatedmetagenomicdata.extract \
    --study LiJ_2017 -o data/lij_2017
```

Once extracted, copy the output directory to HPC storage (if needed) and point
the biomevae training, classification, and pipeline commands at those files.

## Snakemake workflow (recommended)

The recommended way to run the full analysis is the Snakemake workflow under
[`workflow/`](workflow/). It wires `TRAIN → POSTPROCESS → CLASSIFY + FIGURES →
AGGREGATE` into a single DAG, declares the same SLURM resources as the legacy
`hpc/*.slurm` jobs, and only re-runs steps whose outputs are missing — so
resuming a partially-finished study is automatic, no `--skip-*` flags needed.
Two entry points share the same per-stage rule files:

- [`workflow/Snakefile`](workflow/Snakefile) — single-study pipeline.
- [`workflow/Snakefile.meta`](workflow/Snakefile.meta) — meta pipeline that
  loops the same workflow over every multi-label study in the
  `extract-microbiome-data` registry.
- [`workflow/Snakefile.meta.no_interpret`](workflow/Snakefile.meta.no_interpret)
  — meta pipeline variant that skips the SHAP-based interpret steps
  (`postprocess_interpret`, `postprocess_interpret_genus`,
  `biomevae-interpret-compare`) for faster meta-scale runs.

See [`workflow/README.md`](workflow/README.md) for the full reference
(rule list, per-stage resources, single-target re-runs, meta-pipeline study
filters, environment overrides).

### Input layout

`data_root` must contain one sub-directory per study, each holding the three
TSVs produced by `extract-microbiome-data`:

```
<data_root>/
└── LiJ_2017/
    ├── sgb_table.tsv
    ├── phyla.tsv
    └── sample_metadata.tsv
```

Edit [`workflow/config/single_study.yaml`](workflow/config/single_study.yaml)
to set `data_root`, `output_root`, and `study`, or pass them on the CLI with
`--config key=value`.

### Run all single studies with case/control (meta pipeline)

To execute the full single-study workflow (`TRAIN → POSTPROCESS → CLASSIFY +
FIGURES → AGGREGATE`) across **every** multi-label case/control study that
has been extracted under `data_root`, use
[`workflow/Snakefile.meta`](workflow/Snakefile.meta). It forces
`auto_multi_label: true`, so
[`rules/common.smk`](workflow/rules/common.smk) pulls the study list
straight out of `curatedmetagenomicdata.study_registry`, keeping only
studies with `has_case_control=True` and `len(disease_labels) >= min_labels`
(default `2`). Any study whose three TSVs are missing under
`<data_root>/<study>/` is silently skipped, so you can re-run as more
downloads finish.

Edit [`workflow/config/meta_multi_label.yaml`](workflow/config/meta_multi_label.yaml)
to set `data_root` / `output_root`, then:

```bash
# Interactive (login node, biomevae env active)
snakemake -s workflow/Snakefile.meta \
    --configfile workflow/config/meta_multi_label.yaml \
    --cores 16

# SLURM driver (recommended on an HPC) — 1-week walltime
sbatch hpc/run_snakemake_meta.slurm
```

[`hpc/run_snakemake_meta.slurm`](hpc/run_snakemake_meta.slurm) is a
dedicated meta-pipeline driver with `--time=168:00:00` (one week) on
`ei-medium` so it can outlive the per-rule jobs for every study in the
registry. It defaults to `-s workflow/Snakefile.meta
--configfile workflow/config/meta_multi_label.yaml`; any extra flags
you pass are forwarded straight to Snakemake.

Override selection on the CLI instead of editing the YAML:

```bash
snakemake -s workflow/Snakefile.meta --cores 16 \
    --config data_root=/path/to/extracted_studies \
             output_root=/path/to/biomevae_meta_runs \
             label=disease \
             require_case_control=true \
             min_labels=2
```

Useful config knobs (see
[`workflow/config/meta_multi_label.yaml`](workflow/config/meta_multi_label.yaml)):

| Key | Default | Effect |
|---|---|---|
| `require_case_control` | `true` | keep only studies flagged `has_case_control=True` |
| `min_labels` | `2` | minimum number of distinct `disease_labels` |
| `body_site` | *(unset)* | restrict to e.g. `stool` |
| `only_studies` | *(unset)* | explicit allowlist (overrides the filters above) |
| `exclude_studies` | *(unset)* | explicit blocklist |

Each study writes the same `models/` + `figures/` tree as the single-study
pipeline, and a cross-study `<output_root>/meta_summary.tsv` stacking every
`results_summary.tsv` is produced at the end.

#### Skip the SHAP interpret steps (`Snakefile.meta.no_interpret`)

[`workflow/Snakefile.meta.no_interpret`](workflow/Snakefile.meta.no_interpret)
is a drop-in variant of `Snakefile.meta` that disables the SHAP-based
interpretation stage — useful when the per-model SHAP passes dominate
wall-clock time at meta scale and you only need classify / figures /
benchmarking outputs. It sets `skip_interpret: true` in config, which
makes the `aggregate` rule drop its dependency on each model's
`interpret/otu_latent_summary.tsv` and `interpret_genus/otu_latent_summary.tsv`
so Snakemake never pulls the corresponding rules into the DAG.
`hpc/aggregate_results.sh` already no-ops its `biomevae-interpret-compare`
step when fewer than two interpret directories exist on disk.

| Step | Output | `Snakefile.meta` | `Snakefile.meta.no_interpret` |
|---|---|---|---|
| `postprocess_interpret` | `<model>/interpret/otu_latent_summary.tsv` | ✔ | skipped |
| `postprocess_interpret_genus` | `<model>/interpret_genus/otu_latent_summary.tsv` | ✔ | skipped |
| `biomevae-interpret-compare` (inside `aggregate`) | `<study>/figures/interpret_comparison/` | ✔ | skipped |

Launch it the same way as the base meta pipeline:

```bash
# Interactive (login node, biomevae env active)
snakemake -s workflow/Snakefile.meta.no_interpret \
    --configfile workflow/config/meta_multi_label_no_interpret.yaml \
    --cores 16

# Or flip the flag on the base meta Snakefile from the CLI without
# switching entry points:
snakemake -s workflow/Snakefile.meta \
    --configfile workflow/config/meta_multi_label.yaml \
    --cores 16 \
    --config skip_interpret=true
```

Every other knob (`min_labels`, `require_case_control`, `body_site`,
`only_studies`, `exclude_studies`, `eval_seeds`, `extra_args`, `label`,
…) behaves identically to
[`workflow/config/meta_multi_label.yaml`](workflow/config/meta_multi_label.yaml);
the no-interpret variant only differs in the extra `skip_interpret: true`
key in
[`workflow/config/meta_multi_label_no_interpret.yaml`](workflow/config/meta_multi_label_no_interpret.yaml).

### Run on an offline HPC (SLURM driver — recommended)

The lightweight driver job [`hpc/run_snakemake.slurm`](hpc/run_snakemake.slurm)
activates the `biomevae` mamba environment and launches Snakemake in cluster
mode. The driver itself runs on `ei-medium` with a 48 h wall-time so it can
outlive the longest rule jobs; Snakemake then sbatches each per-rule job onto
the right partition (`ei-gpu` for train/postprocess, `ei-short` for
classify/figures/aggregate) using
[`workflow/profiles/slurm/config.yaml`](workflow/profiles/slurm/config.yaml).

```bash
# Single study, all stages, all 10 model variants
sbatch hpc/run_snakemake.slurm \
    -s workflow/Snakefile \
    --config data_root=/path/to/extracted_studies \
             output_root=/path/to/biomevae_runs \
             study=LiJ_2017 \
             label=disease

# Or via the YAML config
sbatch hpc/run_snakemake.slurm \
    -s workflow/Snakefile \
    --configfile workflow/config/single_study.yaml

# Meta pipeline (every multi-label study)
sbatch hpc/run_snakemake.slurm \
    -s workflow/Snakefile.meta \
    --configfile workflow/config/meta_multi_label.yaml
```

Override the mamba install location via `sbatch --export=ALL,MAMBA_EXEC=...,MAMBA_ROOT_PREFIX=...,CONDA_ENV=...`
if your cluster differs from the defaults.

### Run interactively from a login node

If you have already activated the `biomevae` environment on a login node you
can skip the driver job and call Snakemake directly — the SLURM profile still
sbatches each rule:

```bash
snakemake -s workflow/Snakefile \
    --profile workflow/profiles/slurm \
    --configfile workflow/config/single_study.yaml
```

Add `-n` for a dry-run that prints the DAG without sbatching anything.

### Re-run a single step or target a specific output

Snakemake reschedules only the jobs whose outputs are missing or stale, so
resuming after a partial run is automatic. To force a single target, name its
output file:

```bash
sbatch hpc/run_snakemake.slurm \
    -s workflow/Snakefile \
    --configfile workflow/config/single_study.yaml \
    /path/to/biomevae_runs/LiJ_2017/figures/results_summary.tsv
```

### Outputs

Per study, under `<output_root>/<study>/`:

```
LiJ_2017/
├── models/
│   ├── beta-vae/            model.pt + config.json + test/ + embed/ + interpret/ + classify/
│   ├── vanilla-vae/  hyp-vae/  tax-vae/  hyp-tax-vae/  graph-vae/
│   ├── treeprior-vae/  fuse-vae/  tree-dtm-vae/  philrvae/  hyp-philrvae/
│   ├── diva-betavae/  diva-hyp-philrvae/  diva-tree-dtm-vae/
│   ├── phylodiva-betavae/  phylodiva-hyp-philrvae/  phylodiva-tree-dtm-vae/
│   ├── xgboost-baseline/classify/
│   └── aggregate/           all_methods_vs_nmf.json, test_summary.tsv, figures/, slides/
└── figures/                 fig1…fig5.pdf, results_summary.tsv, results_summary.tex
```

Per-rule SLURM stdout/stderr lands in `logs/slurm/<rule>-<jobid>.{out,err}`
relative to the directory you launched Snakemake from (the Snakefile's
`onstart` hook creates the directory automatically).

The legacy `hpc/single_study_pipeline.sh` and the 3-stage shell scripts
documented below still work for ad-hoc submission; the Snakemake workflow is
the recommended way to launch the full analysis on the offline HPC.

## Single-study analysis pipeline

A complete end-to-end HPC pipeline for any single microbiome study,
starting from data downloaded via
[extract-microbiome-data](https://github.com/rvazdev-ex/extract-microbiome-data)
and running VAE training, postprocessing, classification, aggregation,
and publication-ready figure generation.

### Step 1: Extract data (requires internet)

Use the extract-microbiome-data package on a machine with internet access
(e.g. login node or local workstation) to produce `sgb_table.tsv`,
`phyla.tsv`, and `sample_metadata.tsv` for a single study. For example, to
download the LiJ_2017 colorectal cancer study from curatedMetagenomicData:

```bash
python -m curatedmetagenomicdata.extract \
    --study LiJ_2017 -o scratch/lij_2017/data
```

Any single-study directory produced by extract-microbiome-data works — HMP,
MGnify, and custom single-study TSV drops are all supported as long as the
three files are present. Copy the directory to HPC storage if needed.

### Step 2: Run HPC pipeline (no internet required)

By default the pipeline trains with `--epochs 100 --optuna --optuna-trials 100`.
Pass `--extra-args` to override (use `--extra-args ""` to disable). Use
`--study-name` to label SLURM jobs and figure titles and `--label` to pick the
metadata column used for classification (default: `disease`).

Every stage is submitted as parallel SLURM jobs. Run each invocation after
the previous stage's jobs complete:

```bash
# 1. Submit training jobs (one per model variant, uses default extra args)
./hpc/single_study_pipeline.sh \
    --study-name LiJ_2017 \
    --outdir scratch/lij_2017 \
    --data-dir scratch/lij_2017/data

# 2. After training completes, submit postprocessing SLURM jobs
./hpc/single_study_pipeline.sh \
    --study-name LiJ_2017 \
    --outdir scratch/lij_2017 \
    --data-dir scratch/lij_2017/data \
    --skip-train

# 3. After postprocessing completes, submit classification + figure jobs
#    (figure generation is automatically dependency-chained after classification)
./hpc/single_study_pipeline.sh \
    --study-name LiJ_2017 \
    --outdir scratch/lij_2017 \
    --data-dir scratch/lij_2017/data \
    --skip-train --skip-postprocess
```

Or submit the full pipeline as a SLURM job:

```bash
STUDY_NAME=LiJ_2017 \
DATA_DIR=scratch/lij_2017/data \
OUTDIR=scratch/lij_2017 \
  sbatch hpc/single_study_pipeline.slurm
```

The pipeline trains all 10 model variants, extracts embeddings, runs
classification (on the metadata column selected via `--label`, default
`disease`) on each model's embeddings, and generates publication-quality
figures. All stages run as parallel SLURM jobs (classification uses
`ei-short` CPU-only jobs; figure generation is automatically submitted with
a dependency on classification):

| Output | Description |
|--------|-------------|
| `figures/fig1_latent_ordination.pdf` | PCA + UMAP of embeddings coloured by the chosen label |
| `figures/fig2_classification_performance.pdf` | Balanced accuracy, F1, AUROC across models |
| `figures/fig3_confusion_matrices.pdf` | Confusion matrix heatmaps per model |
| `figures/fig4_reconstruction_quality.pdf` | RMSE / MAE reconstruction with NMF baseline |
| `figures/fig5_training_curves.pdf` | Training/validation loss trajectories |
| `figures/results_summary.tsv` | Summary table of all results |
| `figures/results_summary.tex` | LaTeX table (RMSE, MAE, classification metrics) |

The study name passed via `--study-name` (or the `STUDY_NAME` environment
variable for the `.slurm` wrapper) is used in SLURM job labels and in the
titles of every generated figure and LaTeX table, so the same pipeline can
be reused unchanged for any single-study dataset.

## HPC submission (SLURM)

The `hpc/` directory implements a **3-stage pipeline** for general-purpose
model training, plus the end-to-end single-study pipeline described above:
1. parallel model training,
2. parallel per-model post-processing,
3. global aggregation + figure/report generation.

### Prerequisites

- A SLURM-managed HPC cluster (GPU partition for training/post-processing).
- A conda/mamba environment with this package installed.
- Optional taxonomy table (`phyla.tsv`) for taxonomy-aware models.

Environment variables used by the scripts (with defaults):

- `MAMBA_ROOT_PREFIX=/hpc-home/her24bip/.local/share/mamba`
- `MAMBA_EXEC=/hpc-home/her24bip/miniconda3/condabin/mamba`
- `CONDA_ENV=biomevae`

These defaults are defined in:
- `hpc/train_model.sh`
- `hpc/postprocess_model.sh`
- `hpc/aggregate_results.sh`

---

### Stage 1 — Submit all training jobs

```bash
./hpc/submit_all.sh \
  --input /path/to/sgb_table.tsv \
  --taxonomy /path/to/phyla.tsv \
  --outdir /path/to/results
```

Notes:
- `--taxonomy` is optional globally. If omitted, taxonomy-dependent models are skipped.
- One SLURM job is submitted per model; each model writes to `<outdir>/<model-name>/`.
- Extra training flags can be forwarded to every model CLI with `--extra-args`.

Example:

```bash
./hpc/submit_all.sh \
  --input sgb_table.tsv \
  --taxonomy phyla.tsv \
  --outdir runs/hpc \
  --extra-args "--epochs 500 --optuna --optuna-trials 50"
```

Dry run:

```bash
./hpc/submit_all.sh --input sgb_table.tsv --outdir runs/hpc --dry-run
```

Models currently handled by the pipeline:
- `beta-vae`
- `vanilla-vae`
- `hyp-vae`
- `tax-vae` *(requires taxonomy)*
- `hyp-tax-vae` *(requires taxonomy)*
- `graph-vae` *(requires taxonomy)*
- `treeprior-vae` *(requires taxonomy)*
- `fuse-vae` *(requires taxonomy)*
- `tree-dtm-vae` *(requires taxonomy)*
- `philrvae` *(requires taxonomy)*
- `hyp-philrvae` *(requires taxonomy)*
- `diva-betavae` / `diva-hyp-philrvae` / `diva-tree-dtm-vae` *(requires `--domain-col`; taxonomy for the PhILR / tree backbones)*
- `phylodiva-betavae` / `phylodiva-hyp-philrvae` / `phylodiva-tree-dtm-vae` *(requires `--domain-col`; PhILR and tree variants also require taxonomy)*
- `capda-vae` *(single-study; requires taxonomy + metadata label — multi-resolution CLR taxonomy bias + leak-free stratified-K-fold OOF stacking)*

---

### Stage 2 — Submit all post-processing jobs

After training, run:

```bash
./hpc/submit_all_postprocess.sh \
  --input /path/to/sgb_table.tsv \
  --taxonomy /path/to/phyla.tsv \
  --outdir /path/to/results
```

Per model, post-processing executes:
1. `biomevae-test --export`
2. `biomevae-embed --export-recon`
3. `biomevae-interpret`
4. `biomevae-interpret --taxonomy-level genus` *(only if taxonomy was provided)*

Important behavior:
- `flowxformer` is skipped for `biomevae-test` and `biomevae-interpret` in the helper script.
- `hgvae_zi` and `flowxformer` are skipped for `biomevae-interpret`.
- If you pass `--train-jobids`, post-processing jobs are dependency-chained using `--dependency=afterok:<jobid>`.

Example with dependencies:

```bash
./hpc/submit_all_postprocess.sh \
  --input sgb_table.tsv \
  --taxonomy phyla.tsv \
  --outdir runs/hpc \
  --train-jobids 12345,12346,12347,12348,12349,12350,12351,12352,12353,12354
```

---

### Stage 3 — Aggregate all results

Submit aggregation with exported variables (recommended pattern):

```bash
OUTDIR=/path/to/results \
INPUT=/path/to/sgb_table.tsv \
TAXONOMY=/path/to/phyla.tsv \
sbatch hpc/aggregate_results.slurm
```

If no taxonomy is available, set `TAXONOMY=none`.

The aggregation script produces `<outdir>/aggregate/` and runs:
1. test metric collection (`aggregate/test_summary.tsv`),
2. multi-model training-curve plotting,
3. embedding discovery,
4. `biomevae-allcomp`,
5. figure generation (`benchmark`, `violin`, `pairwise`, `hierarchy`, `recon-scatter`),
6. enterosignature figures,
7. benchmark slide deck,
8. cross-model interpretation comparison.

---

### SLURM defaults

Training/post-processing defaults (`hpc/train_model.slurm`, `hpc/postprocess_model.slurm`):

| Resource | Value |
|----------|-------|
| Partition | `ei-gpu` |
| Wall time | 96h (train), 24h (postprocess) |
| CPUs | 20 |
| Memory | 128G |
| GPUs | 1 |

Classification/figures/aggregation defaults (`hpc/classify_model.slurm`, `hpc/generate_figures.slurm`, `hpc/aggregate_results.slurm`):

| Resource | Value |
|----------|-------|
| Partition | `ei-short` |
| Wall time | 2h |
| CPUs | 4 |
| Memory | 16G |
| GPUs | none |

---

### Monitoring jobs

```bash
squeue -u "$USER"                         # list queued/running jobs
scancel <JOB_ID>                           # cancel one job
tail -f runs/hpc/beta-vae/slurm_*.out      # follow training stdout
tail -f runs/hpc/beta-vae/slurm_pp_*.out   # follow postprocess stdout
```
