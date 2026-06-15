# Installation and input format

[← back to main README](README.md)

This page covers how to install `biomevae` and the expected layout of the
input files consumed by every CLI. For pipeline-level reproducibility
contracts (pinned environments, provenance blocks, 5-seed evaluation
protocol) see the [Reproducibility and seeds](README.md#reproducibility-and-seeds)
section of the main README.

## Installation

### Using pip

```bash
pip install -e .                 # development install
# optional extras
pip install -e .[optuna]
pip install -e .[hyper]
pip install -e ".[optuna,hyper,figure,interpret]"
```

### Using conda / mamba

```bash
mamba env create -f requirements.yml
mamba activate biomevae
```

The provided `requirements.yml` installs the base package together with the
optional Optuna- and geoopt-powered features so that all CLI entry points are
ready to use after the environment is created.

### Pinned reference environment

For byte-for-byte reproducible runs, install the pinned environment instead:

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

These are the exact versions (`numpy==1.26.4`, `torch==2.2.2`,
`scikit-learn==1.4.2`, `xgboost==2.0.3`, ...) recorded in the
`metadata['provenance']['packages']` block of every result JSON written
by `biomevae-classify`, `biomevae-allcomp`, `biomevae-nmf`, and related
commands.

## Input format

`--input` is a TSV where **rows are taxa (features)** and columns are:
1) `clade_name` (e.g., `t__SGB123`), 2) `NCBI_tax_id`, 3+) sample columns (numeric).
The code transposes to `[samples × features]`.

For taxonomy-aware training, `--taxonomy` is your `phyla.tsv` mapping `clade_name`
to ranks `k p c o f g s` (TSV by default; header optional). The first column must match
the VAE table’s `clade_name`. Inference commands that need taxonomy (graph, tree, or
phylo-fusion models) accept `--taxonomy` or the `--phyla` alias.
