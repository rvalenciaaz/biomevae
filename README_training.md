# Model training

[← back to main README](README.md)

This page covers every training entry point shipped with `biomevae`, plus
the training-curve plotter and the Optuna hyperparameter-search
integration. All training CLIs are seed-aware and participate in the
5-seed reproducibility contract described in the
[Reproducibility and seeds](README.md#reproducibility-and-seeds) section
of the main README.

- [β-VAE (default CLI)](#β-vae-default-cli)
- [Traditional VAE](#traditional-vae)
- [Taxonomy-aware VAE](#taxonomy-aware-vae)
- [Graph-regularized taxonomy VAE](#graph-regularized-taxonomy-vae)
- [TreeDTM-VAE](#treedtm-vae-tree-dirichlet-tree-multinomial-vae)
- [HG-VAE-ZI (deprecated)](#hierarchical-graph-zi-lognormal-vae-hg-vae-zi-deprecated)
- [Tree-prior VAE](#tree-prior-vae)
- [PhILR-VAE](#philr-vae-phylogenetic-isometric-log-ratio-vae)
- [Hyperbolic PhILR-VAE](#hyperbolic-philr-vae)
- [FlowXFormer (deprecated)](#flowxformer-taxonomy-flow-transformer-deprecated)
- [Phylogenetic fusion VAE](#phylogenetic-fusion-vae)
- [Hyperbolic VAE](#hyperbolic-vae-poincaré-ball)
- [Hyperbolic + Taxonomy-aware](#hyperbolic--taxonomy-aware)
- [DIVA — domain-invariant VAE](#diva--domain-invariant-vae)
- [PhyloDIVA — phylogeny-aware domain adaptation](#phylodiva--phylogeny-aware-domain-adaptation)
- [Plotting training curves](#plotting-training-curves)
- [Optuna hyperparameter search](#optuna-hyperparameter-search)

## β-VAE (default CLI)

```bash
biomevae-train   --input sgb_table.tsv --outdir runs/base   --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64   --log1p --standardize --objective beta   --kl-warmup 150 --beta-max 0.2   --recon mae --early-stop 25
```

## Traditional VAE

```bash
biomevae-train-vanilla   --input sgb_table.tsv --outdir runs/vanilla   --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64   --log1p --standardize   --recon mae --early-stop 25
```

This entry point runs the classical VAE objective with a unit KL weight (β = 1) and no warm-up schedule.

## Taxonomy-aware VAE

```bash
biomevae-train-tax   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/tax   --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64   --log1p --standardize --objective beta --kl-warmup 150 --beta-max 0.2   --recon mae --early-stop 25   --tax-loss-levels g f   --tax-loss-weight 0.2   --tax-laplacian 1e-4 --tax-lap-weights 1.0 0.5 0.25
```

## Graph-regularized taxonomy VAE

```bash
biomevae-train-graph   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/train-graph    --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64   --log1p --standardize --objective beta --kl-warmup 200 --beta-max 0.1   --recon mae --early-stop 25   --lr 1e-3   --gnn gcn --gnn-hidden 64 --gnn-layers 3 --gnn-dropout 0.1
```

`biomevae-train-graph` augments the encoder with a taxonomy graph neural network. Use `--tax-graph-mode branchlen` to weight edges by branch lengths.

## TreeDTM-VAE (tree-Dirichlet-tree-multinomial VAE)

```bash
biomevae-train-tree-dtm   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/tree-dtm-vae   --epochs 200 --batch-size 32 --hidden 128 --latent-dim 8   --encoder-layers 2 --decoder-hidden 256 --decoder-layers 2 --lr 1e-3   --beta-max 1.0 --kl-warmup-frac 0.25   --likelihood dirichlet_tree_multinomial
```

`biomevae-train-tree-dtm` replaces the deprecated TreeNB-VAE family with a
statistically appropriate tree-structured likelihood. The decoder predicts
**local sibling-split probabilities at every internal node** and the
likelihood is a product over internal nodes — so dispersion is learned per
clade rather than per unrelated leaf.

Pick the likelihood appropriate to your input:

- `--likelihood tree_multinomial` — fixed-depth integer counts, no overdispersion.
- `--likelihood dirichlet_tree_multinomial` (default) — overdispersed integer counts; one concentration per sibling group.
- `--likelihood dirichlet_tree` — closed relative-abundance compositions.

See [`docs/tree_dtm_vae_theory.md`](docs/tree_dtm_vae_theory.md) for the
full derivation. The encoder operates on sibling-centred log-ratios at
every internal node and gauges out the within-group scale degree of
freedom by construction.

## ~~Hierarchical graph ZI-LogNormal VAE (HG-VAE-ZI)~~ *(deprecated)*

> **Deprecated.** Use **TreeDTM-VAE** (`biomevae-train-tree-dtm`) instead. HG-VAE-ZI suffers from loss blow-up due to the ZI-LogNormal likelihood's super-exponential mean, a shallow 2-layer SAGEConv encoder, and a post-hoc consistency loss that conflicts with the reconstruction objective. The CLI and model code are retained for reproducibility but are no longer part of the recommended pipeline.

```bash
biomevae-train-hgvae-zi   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/hgvae_zi   --epochs 200 --batch-size 32 --hidden 128 --latent-dim 3 --lr 1e-3   --beta-max 1.0 --beta-warmup-frac 0.3 --lambda-cons 1.0
```

## Tree-prior VAE

```bash
biomevae-train-treeprior   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/tree-prior   --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64   --log1p --standardize --objective beta --kl-warmup 200 --beta-max 0.1   --recon mae --early-stop 25 --lr 1e-3   --prior brownian --prior-sigma 1.0 --branch-reg 0.01
```

The tree-prior CLI adds a structured prior over latent variables derived from the taxonomy tree; tune `--prior-sigma` and `--branch-reg` to control the strength of the phylogenetic regularization.

## PhILR-VAE (phylogenetic isometric log-ratio VAE)

```bash
biomevae-train-philrvae   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/philrvae   --latent-dim 8 --hidden 256 128 --epochs 300 --batch-size 64   --lr 1e-3 --beta-max 1.0 --kl-warmup 100   --pseudocount 0.5 --branchlen-mode unit
```

`biomevae-train-philrvae` uses the Phylogenetic Isometric Log-Ratio (PhILR)
transform — an analytically exact isometric map from the compositional
simplex to R^{p-1} aligned to the taxonomy tree. The encoder and decoder
are simple MLPs operating on the p-1 PhILR coordinates (no transformer, no
per-edge tokens). Euclidean distance in PhILR space equals Aitchison
distance on the simplex, so no geometric or consistency losses are needed.

The likelihood is chosen via `--likelihood` (default `philr_gaussian`):

- `philr_gaussian` — logistic-normal on the simplex (default; for relative-abundance input).
- `multinomial` — fixed-depth integer counts, no overdispersion.
- `dirichlet_multinomial` — globally overdispersed integer counts.
- `dirichlet_tree_multinomial` — clade-overdispersed integer counts (shares the TreeDTM decoder head).
- `dirichlet_tree` — continuous Dirichlet-tree on closed compositions.

The previous independent-leaf Negative-Binomial likelihood is intentionally
removed: it double-models zeros and fights the closure constraint of
compositional data. See [`docs/philrvae_theory.md`](docs/philrvae_theory.md)
for the full derivation.

## Hyperbolic PhILR-VAE

```bash
biomevae-train-hyp-philrvae   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/hyp-philrvae   --latent-dim 8 --hidden 256 128 --epochs 300 --batch-size 64   --lr 1e-3 --beta-max 1.0 --kl-warmup 100   --curvature 1.0 --likelihood philr_gaussian
```

`biomevae-train-hyp-philrvae` places the PhILR-VAE on a Poincaré-ball
latent space. The encoder produces a tangent-space Gaussian, draws are
pushed to the ball via `expmap_0`, and the decoder `logmap_0`-projects the
ball point back to the tangent space **before** the first Euclidean Linear
layer (the audit-D2 fix). The five compositional likelihoods are inherited
from `biomevae-train-philrvae` unchanged. The previous NB / ZINB heads are
removed — see [`docs/hyperbolic_philrvae_theory.md`](docs/hyperbolic_philrvae_theory.md).

Requires the `hyper` extra: `pip install -e .[hyper]`.

## ~~FlowXFormer (taxonomy flow transformer)~~ *(deprecated)*

> **Deprecated.** Use **PhILR-VAE** (`biomevae-train-philrvae`) instead. FlowXFormer suffers from O(E^2) Python-loop precomputation, runs the encoder 3x per training step (main + 2 augmented views for consistency loss), and uses a generic MLP decoder that ignores tree structure. The CLI and model code are retained for reproducibility but are no longer part of the recommended pipeline.

```bash
biomevae-train-flowxformer   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/flowxformer   --latent-dim 8 --hidden 256 128 64 --epochs 200 --batch-size 64   --log1p --standardize --objective beta --kl-warmup 200 --beta-max 0.05   --recon mae --early-stop 25   --d-model 256 --n-layers 6 --n-heads 8   --branchlen-mode unit --uot root_l1 --uot-lambda 0.1
```

## Phylogenetic fusion VAE

```bash
biomevae-train-fuse   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/train-fuse   --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64   --log1p --standardize --objective beta --kl-warmup 50 --beta-max 0.01   --recon mae --early-stop 25 --lr 1e-3   --phylo-embed pca --phylo-embed-dim 32
```

`biomevae-train-fuse` learns an auxiliary phylogenetic embedding (default PCA) and fuses it with the abundance encoder.
When `--standardize` is enabled, the fusion summary still uses the pre-standardized abundances so the phylogeny weights remain well-conditioned.

## Hyperbolic VAE (Poincaré ball)

```bash
biomevae-train-hyp   --input sgb_table.tsv --outdir runs/hyp   --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64   --log1p --standardize --objective beta --kl-warmup 150 --beta-max 0.2   --recon mae --curvature 1.0
```

## Hyperbolic + Taxonomy-aware

```bash
biomevae-train-hyp-tax   --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/hyp_tax   --latent-dim 8 --hidden 128 64 --epochs 200 --batch-size 64 --log1p --standardize   --objective beta --kl-warmup 150 --beta-max 0.2 --recon mae   --tax-loss-levels g f --tax-loss-weight 0.2   --tax-laplacian 1e-4 --tax-lap-weights 1.0 0.5 0.25   --curvature 1.0
```

## DIVA — domain-invariant VAE

DIVA (Ilse et al., MIDL 2020) partitions the latent into three independent
Gaussian factors `z = [z_d ; z_y ; z_x]` with **conditional priors**
`p(z_d | d) = N(μ_d(d), σ_d(d)²)` and `p(z_y | y) = N(μ_y(y), σ_y(y)²)`,
plus a standard prior `p(z_x) = N(0, I)`. Auxiliary classifiers
`q(d | z_d)` and `q(y | z_y)` push each factor to carry the matching side
information. Three backbones are wired up:

```bash
# β-VAE backbone (Gaussian leaves on log-transformed counts)
biomevae-train-diva-beta-vae \
  --input sgb_table.tsv --metadata sample_metadata.tsv \
  --domain-col study --label-col disease \
  --outdir runs/diva-betavae --latent-dim 8 --hidden 128 64 \
  --epochs 200 --batch-size 64 --log1p --standardize --kl-warmup 150

# Hyperbolic PhILR backbone (compositional)
biomevae-train-diva-hyp-philrvae \
  --input sgb_table.tsv --taxonomy phyla.tsv --metadata sample_metadata.tsv \
  --domain-col study --label-col disease \
  --outdir runs/diva-hyp-philrvae --latent-dim 8 --epochs 300 \
  --curvature 1.0 --likelihood philr_gaussian

# TreeDTM backbone (Dirichlet-Tree-Multinomial)
biomevae-train-diva-tree-dtm \
  --input sgb_table.tsv --taxonomy phyla.tsv --metadata sample_metadata.tsv \
  --domain-col study --label-col disease \
  --outdir runs/diva-tree-dtm-vae --latent-dim 8 --epochs 200 \
  --likelihood dirichlet_tree_multinomial
```

Common flags: `--latent-d`, `--latent-y`, `--latent-x` set the per-factor
latent dimensions; `--alpha-d`, `--alpha-y` set the auxiliary
cross-entropy weights. The reconstruction likelihood is the underlying
backbone's; DIVA itself is likelihood-agnostic. See
[`docs/diva_theory.md`](docs/diva_theory.md) for the full ELBO derivation.

## PhyloDIVA — phylogeny-aware domain adaptation

PhyloDIVA augments every DIVA wrapper with three regularisers that close
the gap left by vanilla DIVA on leave-one-study-out (LOSO) microbiome
data:

1. **Gradient-reversed latent study critic** on `z_y` — scrubs study
   fingerprints from the class factor (Ganin & Lempitsky 2015).
2. **Deep CORAL** covariance matching on `z_x` per study (Sun & Saenko 2016).
3. **Phylogenetic smoothness** on the decoder's tree-edge or PhILR-contrast
   outputs (Brownian-motion-style penalty).

```bash
biomevae-train-phylodiva-beta-vae \
  --input sgb_table.tsv --metadata sample_metadata.tsv \
  --domain-col study --label-col disease \
  --outdir runs/phylodiva-betavae \
  --grl-lambda-max 0.5 --coral-weight 0.1

biomevae-train-phylodiva-hyp-philrvae \
  --input sgb_table.tsv --taxonomy phyla.tsv --metadata sample_metadata.tsv \
  --domain-col study --label-col disease \
  --outdir runs/phylodiva-hyp-philrvae \
  --grl-lambda-max 0.5 --coral-weight 0.1 --smooth-weight 0.01

biomevae-train-phylodiva-tree-dtm \
  --input sgb_table.tsv --taxonomy phyla.tsv --metadata sample_metadata.tsv \
  --domain-col study --label-col disease \
  --outdir runs/phylodiva-tree-dtm-vae \
  --grl-lambda-max 0.5 --coral-weight 0.1 --smooth-weight 0.01
```

The GRL critic weight follows the canonical DANN sigmoid ramp
`λ(t) = λ_max·(2/(1+exp(-γt))-1)` (default γ = 10), so the adversary is
off at the start of training and approaches `--grl-lambda-max` at the end.
See [`docs/phylodiva_theory.md`](docs/phylodiva_theory.md) for the full
loss and diagnostic expectations.

## Plotting training curves

Every training CLI writes `<outdir>/training_log.tsv`, which includes per-epoch metrics such as `train_loss`, `val_loss`, `train_recon`, `val_recon`, `train_kld`, `val_kld`, and `val_r2`. Use `biomevae-plot-training-curves` to visualize **all models at once**, similarly to how `biomevae-allcomp` compares methods. The script generates three plots for a chosen metric: (1) training-only curves, (2) validation-only curves, and (3) training + validation overlaid.

```bash
biomevae-plot-training-curves \
  --log base=runs/base/training_log.tsv \
  --log vanilla=runs/vanilla/training_log.tsv \
  --log tax=runs/tax/training_log.tsv \
  --log graph=runs/train-graph/training_log.tsv \
  --log treeprior=runs/tree-prior/training_log.tsv \
  --log fuse=runs/train-fuse/training_log.tsv \
  --log hyp=runs/hyp/training_log.tsv \
  --log hyp_tax=runs/hyp_tax/training_log.tsv \
  --log tree-dtm-vae=runs/tree-dtm-vae/training_log.tsv \
  --log philrvae=runs/philrvae/training_log.tsv \
  --log hyp-philrvae=runs/hyp-philrvae/training_log.tsv \
  --log diva-tree-dtm=runs/diva-tree-dtm-vae/training_log.tsv \
  --log phylodiva-tree-dtm=runs/phylodiva-tree-dtm-vae/training_log.tsv \
  --metric loss \
  --output runs/training_curves \
  --title "Training comparison"
```

Each `--log` entry is `NAME=PATH`, where `NAME` becomes the legend label for that model. The script saves `training_<metric>_curves.png`, `validation_<metric>_curves.png`, and `train_val_<metric>_curves.png` in the output directory (default: current directory). Use `--show` to display the plots interactively after saving. The plotting utilities require matplotlib, which is available via the optional `figure` extra: `pip install -e .[figure]`.

The y-axis defaults to a log scale (`--yscale log`) so that compositional-NLL based models — **PhILR-VAE**, **Hyperbolic PhILR-VAE** and **TreeDTM-VAE** whose Dirichlet- / multinomial-NLL reconstruction loss sits in the hundreds / thousands — stay visible alongside MSE-based VAEs whose reconstruction loss sits near zero. Pass `--yscale linear` if you prefer a linear axis (e.g. when plotting a single model family). Logs that are missing the requested metric columns are skipped with a warning rather than failing the whole run, so older training logs do not break recon-curve aggregation. Use `--metric recon` (or call the script twice, once per metric) to obtain the stationary reconstruction curve that is the only convergence diagnostic comparable across epochs for β-annealed VAEs. The HPC aggregation step (`hpc/aggregate_results.sh`) now plots both `loss` and `recon` curves automatically.

## Optuna hyperparameter search

All training entry points (`biomevae-train`, `biomevae-train-vanilla`, `biomevae-train-tax`, `biomevae-train-graph`, `biomevae-train-tree-dtm`, `biomevae-train-treeprior`, `biomevae-train-philrvae`, `biomevae-train-hyp-philrvae`, `biomevae-train-fuse`, `biomevae-train-hyp`, `biomevae-train-hyp-tax`, and the DIVA / PhyloDIVA wrappers: `biomevae-train-diva-beta-vae`, `biomevae-train-diva-hyp-philrvae`, `biomevae-train-diva-tree-dtm`, `biomevae-train-phylodiva-beta-vae`, `biomevae-train-phylodiva-hyp-philrvae`, `biomevae-train-phylodiva-tree-dtm`) include an Optuna integration to automate hyperparameter sweeps. Install the optional dependency group first:

```bash
pip install -e .[optuna]
```

### Basic usage

Append `--optuna` to any training command to launch a study. The example below runs 100 trials of the taxonomy-aware hyperbolic model and saves intermediate artifacts in `<outdir>/optuna_trials/`:

```bash
biomevae-train --input sgb_table.tsv --outdir runs/base   --optuna --optuna-trials 100
biomevae-train-vanilla --input sgb_table.tsv --outdir runs/vanilla   --optuna --optuna-trials 100
biomevae-train-tax --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/tax   --optuna --optuna-trials 100
biomevae-train-graph --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/train-graph   --optuna --optuna-trials 100
biomevae-train-tree-dtm --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/tree-dtm-vae   --optuna --optuna-trials 100
biomevae-train-treeprior --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/tree-prior   --optuna --optuna-trials 100
biomevae-train-philrvae --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/philrvae   --optuna --optuna-trials 100
biomevae-train-hyp-philrvae --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/hyp-philrvae   --optuna --optuna-trials 100
biomevae-train-fuse --input sgb_table.tsv --taxonomy phyla.tsv --outdir runs/train-fuse   --optuna --optuna-trials 100
biomevae-train-hyp --input sgb_table.tsv --outdir runs/hyp   --optuna --optuna-trials 100
biomevae-train-hyp-tax --input sgb_table.tsv --taxonomy phyla.tsv   --outdir runs/hyp_tax   --optuna --optuna-trials 100
biomevae-train-diva-tree-dtm --input sgb_table.tsv --taxonomy phyla.tsv --metadata sample_metadata.tsv --domain-col study --label-col disease   --outdir runs/diva-tree-dtm-vae --optuna --optuna-trials 100
biomevae-train-phylodiva-tree-dtm --input sgb_table.tsv --taxonomy phyla.tsv --metadata sample_metadata.tsv --domain-col study --label-col disease   --outdir runs/phylodiva-tree-dtm-vae --optuna --optuna-trials 100
```

Each trial directory stores the configuration used for that run. After the study finishes, the best set of parameters is retrained and written to `optuna_best_params.json`; a CSV summary of all trials is exported to `optuna_trials.csv`.

### Custom search spaces

By default the search explores a set of sensible hyperparameters defined in code. You can override or extend the search space with a JSON configuration passed via `--optuna-config`:

```bash
biomevae-train   --input sgb_table.tsv --outdir runs/base_opt   --optuna --optuna-config configs/optuna_search_space.json
```

Use the provided template at [`configs/optuna_search_space.template.json`](configs/optuna_search_space.template.json) as a starting point. Copy it to a new file, adjust the ranges, and point `--optuna-config` to the edited copy. Keys follow the Optuna `trial.suggest_*` API—set `"method"` to the suggestion method name and provide its keyword arguments. Nested parameters (for example, `model_kwargs.curvature`) use dot notation. Values without a `method` field are treated as constants and will override the defaults.
