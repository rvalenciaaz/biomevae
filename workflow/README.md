# biomevae Snakemake workflow

End-to-end Snakemake orchestration of the biomevae analysis. Two entry
points share the same per-stage rule files:

```
workflow/
├── Snakefile                       # single-study pipeline
├── Snakefile.meta                  # meta pipeline across every multi-label study
├── Snakefile.meta.no_interpret     # meta pipeline variant that skips the interpret steps
├── rules/
│   ├── common.smk         # model catalogue, path helpers, study resolver
│   ├── train.smk          # Step 1 – per-model training
│   ├── postprocess.smk    # Step 2 – test / embed / interpret(×2)
│   ├── classify.smk       # Step 3 – VAE + XGBoost baseline classification
│   ├── figures.smk        # Step 4 – publication figures + summary tables
│   └── aggregate.smk      # Step 5 – cross-model benchmarking (honours `skip_interpret`)
├── scripts/
│   └── no_mamba_wrapper.sh
├── profiles/
│   └── slurm/             # SLURM cluster profile (ei-gpu / ei-short)
│       └── config.yaml
└── config/
    ├── single_study.yaml
    ├── meta_multi_label.yaml
    └── meta_multi_label_no_interpret.yaml
```

## Pipeline stages

Each stage is a modular Snakemake rule. Snakemake's DAG wires them
together automatically, so you launch the whole thing once and it
runs `TRAIN → POSTPROCESS → CLASSIFY + FIGURES → AGGREGATE`:

| Stage | Rule(s) | Produces |
|---|---|---|
| Train | `train_model` (×10 models) | `<study>/models/<model>/model.pt`, `config.json`, `training_log.tsv` |
| Postprocess | `postprocess_test`, `postprocess_embed`, `postprocess_interpret`, `postprocess_interpret_genus` | `<model>/test/`, `<model>/embed/`, `<model>/interpret/`, `<model>/interpret_genus/` |
| Classify | `classify` (×10), `classify_xgboost_baseline` | `<model>/classify/classification_results.json`, `models/xgboost-baseline/classify/…` |
| Figures | `single_study_figures` | `<study>/figures/fig1…fig5.pdf`, `results_summary.tsv` |
| Aggregate | `aggregate` | `<study>/models/aggregate/{all_methods_vs_nmf.json,test_summary.tsv,figures/…,slides/…}` |

Re-run any single step by asking Snakemake for its output:

```bash
snakemake -s workflow/Snakefile --cores 4 \
    --configfile workflow/config/single_study.yaml \
    /path/to/output/LiJ_2017/models/tax-vae/classify/classification_results.json
```

## Inputs

`data_root` must point at a directory populated by the
[extract-microbiome-data](https://github.com/rvazdev-ex/extract-microbiome-data)
package, with one sub-directory per study:

```
<data_root>/
├── LiJ_2017/
│   ├── sgb_table.tsv
│   ├── phyla.tsv
│   └── sample_metadata.tsv
├── QinJ_2012/
│   ├── sgb_table.tsv
│   ├── phyla.tsv
│   └── sample_metadata.tsv
└── …
```

Generate that layout with, for example:

```bash
python -m curatedmetagenomicdata.extract \
    --all-studies --case-control-only --separate \
    -o /data/extracted_studies
```

## Single-study pipeline

Train all ten biomevae variants, run every postprocess + classify step
and produce the full figure set for a single study:

```bash
snakemake -s workflow/Snakefile --cores 8 \
    --configfile workflow/config/single_study.yaml
```

or override config on the CLI:

```bash
snakemake -s workflow/Snakefile --cores 8 \
    --config data_root=/data/extracted_studies \
             output_root=/scratch/biomevae_runs \
             study=LiJ_2017 \
             label=disease
```

Outputs land under `<output_root>/<study>/`:

```
<output_root>/LiJ_2017/
├── models/
│   ├── beta-vae/            model.pt + config.json + test/ + embed/ + interpret/ + classify/
│   ├── vanilla-vae/
│   ├── hyp-vae/
│   ├── tax-vae/
│   ├── hyp-tax-vae/
│   ├── graph-vae/
│   ├── treeprior-vae/
│   ├── fuse-vae/
│   ├── tree-dtm-vae/
│   ├── philrvae/
│   ├── hyp-philrvae/       # Hyperbolic PhILR-NB VAE (dense)
│   ├── hyp-philr-zinb/     # Hyperbolic PhILR-ZINB VAE (sparsity-aware)
│   ├── capda-vae/          # single-study CAPDA-VAE (CLR taxonomy + OOF stacking)
│   ├── xgboost-baseline/classify/
│   └── aggregate/           all_methods_vs_nmf.json, test_summary.tsv, figures/, slides/
└── figures/                 fig1…fig5.pdf, results_summary.tsv
```

## Meta pipeline (all multi-label studies)

`Snakefile.meta` is the single-study pipeline wrapped in a study
loop.  It defaults to `auto_multi_label: true`, which pulls the list
of *multi-label* studies (at least two distinct `disease_labels` **and**
`has_case_control = True`) straight out of
`curatedmetagenomicdata.study_registry`.  Any study whose three TSVs
are missing under `data_root` is silently skipped, so you can keep
re-running the pipeline as more downloads finish.

```bash
snakemake -s workflow/Snakefile.meta --cores 16 \
    --configfile workflow/config/meta_multi_label.yaml
```

When it completes you get, per study, the same tree as the
single-study pipeline, plus one cross-study `meta_summary.tsv` at the
root of `output_root` that stacks every study's `results_summary.tsv`.

Tune the study selection via config:

```yaml
auto_multi_label: true
min_labels: 3              # require ≥3 classes
require_case_control: true
body_site: stool           # only gut studies
only_studies:              # allowlist (overrides the above)
  - LiJ_2017
  - HMP_2019_ibdmdb
exclude_studies:           # blocklist
  - SomeStudy
```

### Skipping the interpret steps

`Snakefile.meta.no_interpret` is a drop-in variant of `Snakefile.meta`
that disables the SHAP-based interpretation stage:

| Step | Output | `Snakefile.meta` | `Snakefile.meta.no_interpret` |
|---|---|---|---|
| `postprocess_interpret` | `<model>/interpret/otu_latent_summary.tsv` | ✔ | skipped |
| `postprocess_interpret_genus` | `<model>/interpret_genus/otu_latent_summary.tsv` | ✔ | skipped |
| `biomevae-interpret-compare` (inside `aggregate`) | `<study>/figures/interpret_comparison/` | ✔ | skipped |

Use it when the per-model SHAP passes dominate wall-clock time and you
only need the classify / figures / benchmarking outputs at meta scale:

```bash
snakemake -s workflow/Snakefile.meta.no_interpret --cores 16 \
    --configfile workflow/config/meta_multi_label_no_interpret.yaml
```

Mechanism: the variant sets `skip_interpret: true` in config, which
makes the `aggregate` rule drop its dependency on the two
`<model>/interpret*/otu_latent_summary.tsv` TSVs so Snakemake never
pulls the corresponding rules into the DAG.
`hpc/aggregate_results.sh` already no-ops its `biomevae-interpret-compare`
step when fewer than two interpret directories exist on disk, so no
shell changes are needed.

Every other knob (`min_labels`, `require_case_control`, `body_site`,
`only_studies`, `exclude_studies`, `eval_seeds`, `extra_args`, `label`,
…) behaves identically to the base meta pipeline. You can also flip the
flag on the base meta Snakefile from the CLI without switching entry
points:

```bash
snakemake -s workflow/Snakefile.meta --cores 16 \
    --configfile workflow/config/meta_multi_label.yaml \
    --config skip_interpret=true
```

## Running on an HPC cluster

Every rule declares the same SLURM resources as the legacy
`hpc/*.slurm` jobs, and a ready-to-use SLURM profile lives at
`workflow/profiles/slurm/config.yaml`:

| Stage | Partition | Time | CPUs | Memory | GPU |
|---|---|---|---|---|---|
| `train_model` | `ei-gpu` | 96 h | 20 | 128 G | `--gres=gpu:1` |
| `postprocess_{test,embed,interpret,interpret_genus}` | `ei-gpu` | 24 h | 20 | 128 G | `--gres=gpu:1` |
| `classify`, `classify_xgboost_baseline` | `ei-short` | 2 h | 4 | 16 G | – |
| `single_study_figures` | `ei-short` | 2 h | 4 | 16 G | – |
| `aggregate` | `ei-short` | 2 h | 4 | 16 G | – |
| `meta_summary` | `ei-short` | 30 m | 1 | 4 G | – |

These numbers mirror `hpc/train_model.slurm`, `hpc/postprocess_model.slurm`,
`hpc/classify_model.slurm`, `hpc/generate_figures.slurm` and
`hpc/aggregate_results.slurm` exactly, so the Snakemake pipeline is a
drop-in replacement for the existing SLURM shell-script pipeline.

### One-shot driver job (recommended)

Submit a single lightweight driver job that activates the `biomevae`
mamba environment and runs Snakemake in cluster mode – the driver
schedules every per-rule sbatch job for you. The driver itself runs on
`ei-medium` with a 48 h wall-time so it can outlive the longest rule
jobs (training asks for 96 h on `ei-gpu`, so a 2 h `ei-short` slot is
not enough and SLURM would park the driver with `(PartitionTimeLimit)`):

```bash
sbatch hpc/run_snakemake.slurm \
    -s workflow/Snakefile \
    --configfile workflow/config/single_study.yaml
```

Meta pipeline over every multi-label study (use the dedicated driver
`hpc/run_snakemake_meta.slurm`, which has a one-week walltime
(`--time=168:00:00`) so it can outlive the per-rule jobs for every
study in the registry):

```bash
sbatch hpc/run_snakemake_meta.slurm
```

The meta driver defaults to `-s workflow/Snakefile.meta --configfile
workflow/config/meta_multi_label.yaml`; extra flags are still forwarded
to Snakemake if you need to override config or target a specific
output.

Everything after the script name is forwarded straight to Snakemake,
so you can override config from the CLI or target a single output:

```bash
sbatch hpc/run_snakemake.slurm \
    -s workflow/Snakefile \
    --config data_root=/ei/projects/.../extracted \
             output_root=/ei/scratch/.../biomevae_runs \
             study=LiJ_2017

sbatch hpc/run_snakemake.slurm \
    -s workflow/Snakefile \
    --configfile workflow/config/single_study.yaml \
    -- /ei/scratch/.../biomevae_runs/LiJ_2017/models/tax-vae/model.pt
```

`hpc/run_snakemake.slurm` reads three optional environment variables
(defaults match the existing `hpc/*.sh` scripts):

| Variable | Default |
|---|---|
| `MAMBA_EXEC` | `/hpc-home/her24bip/miniconda3/condabin/mamba` |
| `MAMBA_ROOT_PREFIX` | `/hpc-home/her24bip/.local/share/mamba` |
| `CONDA_ENV` | `biomevae` |

Override them with `sbatch --export=…` if your installation differs.

### Interactive launch from a login node

If you're already inside an activated `biomevae` environment on a
login node you can skip the driver job and call Snakemake directly –
it will sbatch each rule with the same resources:

```bash
snakemake -s workflow/Snakefile \
    --configfile workflow/config/single_study.yaml \
    --profile workflow/profiles/slurm
```

SLURM stdout/stderr for every per-rule job lands in
`logs/slurm/<rule>-<jobid>.{out,err}` relative to the directory you
launched Snakemake from (the Snakefile's `onstart` hook creates the
directory automatically).

The legacy HPC shell scripts under `hpc/` still work for ad-hoc
submission; the Snakemake workflow is now the recommended way to
launch the full analysis and matches their SLURM resource profile
exactly.

## Environment

The rules call the `biomevae-*` CLIs directly, so run Snakemake from
inside an activated `biomevae` environment (conda/mamba).  The
aggregate rule reuses the existing `hpc/aggregate_results.sh` via the
tiny `scripts/no_mamba_wrapper.sh` stub – it strips the `mamba run -n
<env>` prefix so every CLI call in that script runs inside whatever
environment Snakemake itself was launched from.
