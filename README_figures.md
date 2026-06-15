# Figures, diagnostics, and theory documentation

[← back to main README](README.md)

This page covers every figure-generating CLI in `biomevae`
(benchmark bars, ordinations, enterosignatures, slide decks,
reconstruction scatter plots, violins, per-level breakdowns,
pairwise significance heatmaps, cross-model interpretation
comparisons, UMAP ordinations, posterior-collapse diagnostics,
and sparsity-aware metrics) plus a pointer to the mathematical theory
write-ups in `docs/`. All statistical outputs (including the pairwise
p-value heatmaps) rely on the 5-seed evaluation protocol described in the
[Reproducibility and seeds](README.md#reproducibility-and-seeds) section
of the main README.

- [Benchmark figure generation](#benchmark-figure-generation)
- [Benchmark enterosignature figures](#benchmark-enterosignature-figures)
- [Benchmark slide deck generation](#benchmark-slide-deck-generation)
- [Reconstruction scatter plots](#reconstruction-scatter-plots)
- [Reconstruction error distributions](#reconstruction-error-distributions)
- [Per-taxonomy-level metric breakdown](#per-taxonomy-level-metric-breakdown)
- [Pairwise statistical significance tables](#pairwise-statistical-significance-tables)
- [Cross-model interpretation comparison](#cross-model-interpretation-comparison)
- [UMAP ordinations](#umap-ordinations)
- [Posterior collapse diagnostics](#posterior-collapse-diagnostics)
- [Sparsity-aware reconstruction metrics](#sparsity-aware-reconstruction-metrics)
- [Mathematical documentation](#mathematical-documentation)

## Benchmark figure generation

Use `biomevae-benchmark-figure` to turn the reconstruction summaries emitted by
`biomevae-comparetonmf` or `biomevae-allcomp` into publication-ready bar charts.
Pass one or more JSON files whose top-level keys are method names and whose
values mirror the structure in `runs/all_methods_vs_nmf.json`:

```bash
biomevae-benchmark-figure   --input runs/all_methods_vs_nmf.json \
                          --metric rmse --metric mae \
                          --title "Reconstruction metrics" --baseline nmf --matrix sgb_table.tsv \
                          --embedding base=runs/base/embed_eval/embeddings.tsv \
                          --embedding vanilla=runs/vanilla/embed_eval/embeddings.tsv \
                          --embedding tax=runs/tax/embed_eval/embeddings.tsv \
                          --embedding graph=runs/train-graph/embed_eval/embeddings.tsv \
                          --embedding treeprior=runs/tree-prior/embed_eval/embeddings.tsv \
                          --embedding tree-dtm-vae=runs/tree-dtm-vae/embed_eval/embeddings.tsv \
                          --embedding philrvae=runs/philrvae/embed_eval/embeddings.tsv \
                          --embedding fuse=runs/train-fuse/embed_eval/embeddings.tsv \
                          --embedding hyp=runs/hyp/embed_eval/embeddings.tsv \
                          --embedding hyp_tax=runs/hyp_tax/embed_eval/embeddings.tsv \
                          --output figures/benchmark_metrics.pdf \
                          --ordinations-output figures/benchmark_ordinations.pdf
```

To save PNGs instead of PDFs, keep the same multi-model setup and change the
output extensions:

```bash
biomevae-benchmark-figure   --input runs/all_methods_vs_nmf.json \
                          --metric rmse --metric mae \
                          --title "Reconstruction metrics" --baseline nmf --matrix sgb_table.tsv \
                          --embedding base=runs/base/embed_eval/embeddings.tsv \
                          --embedding vanilla=runs/vanilla/embed_eval/embeddings.tsv \
                          --embedding tax=runs/tax/embed_eval/embeddings.tsv \
                          --embedding graph=runs/train-graph/embed_eval/embeddings.tsv \
                          --embedding treeprior=runs/tree-prior/embed_eval/embeddings.tsv \
                          --embedding tree-dtm-vae=runs/tree-dtm-vae/embed_eval/embeddings.tsv \
                          --embedding philrvae=runs/philrvae/embed_eval/embeddings.tsv \
                          --embedding fuse=runs/train-fuse/embed_eval/embeddings.tsv \
                          --embedding hyp=runs/hyp/embed_eval/embeddings.tsv \
                          --embedding hyp_tax=runs/hyp_tax/embed_eval/embeddings.tsv \
                          --output figures/benchmark_metrics.png \
                          --ordinations-output figures/benchmark_ordinations.png
```

The CLI highlights the `--baseline` method (default: `nmf`) and accepts optional
`--rename old=new` assignments to relabel methods before plotting. Control the
rendered size via `--figsize WIDTHxHEIGHT` (in inches). When `--output` is
omitted the figure is displayed interactively; otherwise the image is written to
disk and the absolute path is printed. Matplotlib must be installed to generate
the plot (the dependency is included in `requirements.yml` or installable via
`pip install -e .[figure]`). Supplying
`--matrix` computes PCA and t-SNE ordinations from the counts matrix (use
`--no-matrix-log1p` to disable the default log transform). Provide
`--embedding NAME=PATH` to add latent-space embeddings—repeat the flag to
contrast several models using the same `NAME=PATH` syntax as
`biomevae-allcomp`. Each metric is rendered as an independent figure; omitting
`--metric` plots every score shared by the loaded summaries. Ordinations are
written to a dedicated grid via `--ordinations-output`, ensuring the metric
figures stay focused on the bar charts.

## Benchmark enterosignature figures

Use `biomevae-benchmark-figures-enterosignatures` to generate the same benchmark
metric figures and ordinations as `biomevae-benchmark-figure`, while also
coloring the ordinations by enterosignatures inferred from genus-level
abundances. The tool aggregates the counts matrix to genus level, computes
Bray-Curtis distances, and clusters samples with partitioning around medoids
(PAM) to assign enterosignatures. It additionally compares how closely each
embedding's own PAM assignment agrees with the genus-level enterosignatures via
the adjusted Rand index:

```bash
biomevae-benchmark-figures-enterosignatures   --input runs/all_methods_vs_nmf.json \
                                            --metric rmse --metric mae \
                                            --title "Reconstruction metrics" \
                                            --baseline nmf \
                                            --matrix sgb_table.tsv \
                                            --taxonomy phyla.tsv \
                                            --embedding base=runs/base/embed_eval/embeddings.tsv \
                                            --embedding vanilla=runs/vanilla/embed_eval/embeddings.tsv \
                                            --embedding tax=runs/tax/embed_eval/embeddings.tsv \
                                            --embedding graph=runs/train-graph/embed_eval/embeddings.tsv \
                                            --embedding treeprior=runs/tree-prior/embed_eval/embeddings.tsv \
                                            --embedding tree-dtm-vae=runs/tree-dtm-vae/embed_eval/embeddings.tsv \
                                            --embedding philrvae=runs/philrvae/embed_eval/embeddings.tsv \
                                            --embedding fuse=runs/train-fuse/embed_eval/embeddings.tsv \
                                            --embedding hyp=runs/hyp/embed_eval/embeddings.tsv \
                                            --embedding hyp_tax=runs/hyp_tax/embed_eval/embeddings.tsv \
                                            --clusters 2 \
                                            --alpha-range 0-30 \
                                            --bicross-folds 4 \
                                            --bicross-repetitions 20 \
                                            --output figures/benchmark_metrics.pdf \
                                            --ordinations-output figures/benchmark_ordinations.pdf \
                                            --enterosignature-output figures/benchmark_enterosignatures.pdf \
                                            --rank-selection-output enterosignatures_rank_selection.pdf \
                                            --comparison-output figures/embedding_enterosignature_agreement.pdf \
                                            --agreement-output figures/embedding_enterosignature_ari.pdf \
                                            --geometry-plot-output figures/enterosignature_geometry.pdf \
                                            --procrustes-output figures/enterosignature_procrustes.pdf \
                                            --contingency-plot-output figures/enterosignature_contingency.pdf
```

## Benchmark slide deck generation

For talks or internal reports, `biomevae-benchmark-slides` produces a minimal
Beamer deck containing benchmark figures, tables, and optional ordination
panels. The command reuses the same JSON inputs and metrics as the figure CLI:

```bash
biomevae-benchmark-slides   --input runs/all_methods_vs_nmf.json \
                          --metric rmse --metric mae \
                          --title "Benchmark Overview" \
                          --subtitle "Hold-out reconstruction" \
                          --author "biomevae team" \
                          --matrix sgb_table.tsv \
                          --embedding base=runs/base/embed_eval/embeddings.tsv \
                          --embedding vanilla=runs/vanilla/embed_eval/embeddings.tsv \
                          --embedding tax=runs/tax/embed_eval/embeddings.tsv \
                          --embedding graph=runs/train-graph/embed_eval/embeddings.tsv \
                          --embedding treeprior=runs/tree-prior/embed_eval/embeddings.tsv \
                          --embedding tree-dtm-vae=runs/tree-dtm-vae/embed_eval/embeddings.tsv \
                          --embedding philrvae=runs/philrvae/embed_eval/embeddings.tsv \
                          --embedding fuse=runs/train-fuse/embed_eval/embeddings.tsv \
                          --embedding hyp=runs/hyp/embed_eval/embeddings.tsv \
                          --embedding hyp_tax=runs/hyp_tax/embed_eval/embeddings.tsv \
                          --figure-output slides/benchmark_figure.pdf \
                          --ordinations-output slides/benchmark_ordinations.pdf \
                          --slides-output slides/benchmark_slides.tex
```

`--figure-output` acts as the base path for the generated figures; when multiple
metrics are plotted the tool appends the metric name to the filename. The path
is linked on each slide so viewers can download the PDF. `--slides-output`
stores the generated LaTeX file (run `pdflatex` to obtain the presentation).
The CLI supports the same renaming logic as the figure tool, plus customization
hooks for the Beamer theme, title-page metadata, and the frame/table headings
via `--theme`, `--subtitle`, `--author`, `--frame-title`, and `--table-title`.
Supplying `--matrix` enables PCA/t-SNE/UMAP ordinations for the counts matrix,
while `--embedding NAME=PATH` adds latent spaces to the ordination grid. The
resulting deck includes a dedicated slide for the ordinations alongside the
per-metric figures and tables.

## Reconstruction scatter plots

Use `biomevae-recon-scatter` to generate observed vs predicted scatter plots from
model reconstructions. Each model's `recon.tsv` (produced by `biomevae-test --export`)
is plotted against the original counts matrix:

```bash
biomevae-recon-scatter \
    --input sgb_table.tsv \
    --recon base=runs/base/test/recon.tsv \
    --recon tax=runs/tax/test/recon.tsv \
    --recon tree-dtm-vae=runs/tree-dtm-vae/test/recon.tsv \
    --recon philrvae=runs/philrvae/test/recon.tsv \
    --output figures/recon_scatter.pdf \
    --sample-frac 0.05
```

Each panel shows a subsample of (observed, predicted) pairs with an identity line
overlay and annotated R² and RMSE. Use `--no-log1p` to disable the default log
transform. Adjust `--sample-frac` to control the number of plotted points.

## Reconstruction error distributions

Use `biomevae-recon-violin` to visualise the per-fold error distributions from a
cross-validation benchmark as violin plots with overlaid strip points:

```bash
biomevae-recon-violin \
    --input runs/all_methods_vs_nmf.json \
    --metric rmse --metric mae --metric r2 \
    --baseline nmf \
    --output figures/benchmark_violin.pdf
```

Each metric produces an independent figure. Methods are sorted by their median
value, and the baseline is highlighted in gray. The violin bodies show the
distribution shape; individual fold scores are plotted as jittered points.

## Per-taxonomy-level metric breakdown

Use `biomevae-hierarchy-figure` to create grouped bar charts that break down
reconstruction metrics by taxonomy level. This requires the hierarchy-aware
metrics produced by `biomevae-allcomp --taxonomy`:

```bash
biomevae-hierarchy-figure \
    --input runs/all_methods_vs_nmf.json \
    --metric rmse --metric mae \
    --levels phylum class order family genus species \
    --baseline nmf \
    --output figures/hierarchy_metrics.pdf
```

Each metric generates a figure with one group of bars per taxonomy level and one
bar per method within each group.

## Pairwise statistical significance tables

Use `biomevae-pairwise-table` to export the pairwise sign-test comparisons between
methods as TSV tables, LaTeX tables, and p-value heatmaps:

```bash
biomevae-pairwise-table \
    --input runs/all_methods_vs_nmf.json \
    --metric rmse --metric mae \
    --output figures/pairwise \
    --format both
```

For each metric, this produces:
- `{output}_{metric}_pairwise.tsv` — full comparison table
- `{output}_{metric}_pairwise.tex` — LaTeX-formatted table (ready for papers)
- `{output}_{metric}_pairwise.pdf` — heatmap of BH-adjusted p-values with significance stars

By default the CLI runs `--test seed`, which dispatches the paired
Wilcoxon signed-rank test and the Nadeau–Bengio corrected paired
t-test on the per-seed mean metrics embedded in
`metadata['per_seed_mean_metrics']` (5 seeds by default — see the
[Reproducibility and seeds](README.md#reproducibility-and-seeds)
section). The canonical `p_value` column is the Nadeau–Bengio
corrected t value; additional columns (`p_value_sign`,
`p_value_wilcoxon`, `p_value_tcorrected`) are reported side-by-side
in the seed-level TSV output. Pass `--test fold` to fall back to the
legacy sign test on pooled fold metrics (only meaningful when every
method was evaluated on identical fold partitions). The
`--train-fraction` flag (default `0.9`) must match the train-set
fraction used during cross-validation so that the Nadeau–Bengio
variance inflation reflects the actual train/test split ratio.

## Cross-model interpretation comparison

Use `biomevae-interpret-compare` to compare feature-importance rankings across
multiple models. Point each `--interpret-dir` to a directory produced by
`biomevae-interpret`:

```bash
biomevae-interpret-compare \
    --interpret-dir base=runs/base/interpret \
    --interpret-dir tax=runs/tax/interpret \
    --interpret-dir tree-dtm-vae=runs/tree-dtm-vae/interpret \
    --interpret-dir philrvae=runs/philrvae/interpret \
    --top-k 20 \
    --output figures/interpret_comparison
```

This generates:
- `{output}_consensus_heatmap.pdf` — heatmap of normalised |SHAP| values for
  the top features across models
- `{output}_rank_agreement.pdf` — model × model heatmap of Spearman rank correlation
- `{output}_consensus_features.tsv` — features that appear in the top-k of
  multiple models, with average importance scores

## UMAP ordinations

When the `umap-learn` package is installed (included in the `figure` optional
extra), all ordination outputs automatically include UMAP projections alongside
PCA and t-SNE. The ordination grid adds a third column for UMAP when available.
No CLI flags are required—UMAP is detected and used automatically.

## Posterior collapse diagnostics

The training loop now logs per-dimension KL divergence values (`kl_dim_0`,
`kl_dim_1`, ...) and the count of active latent units (`active_units`, defined
as dimensions with mean KL > 0.01) in `training_log.tsv`. These columns enable
plotting posterior collapse diagnostics using `biomevae-plot-training-curves`:

```bash
biomevae-plot-training-curves \
    --log base=runs/base/training_log.tsv \
    --log tree-dtm-vae=runs/tree-dtm-vae/training_log.tsv \
    --metric active_units \
    --output runs/training_curves \
    --title "Active latent units"
```

## Sparsity-aware reconstruction metrics

The reconstruction metric suite now includes zero/nonzero precision and recall
in addition to the standard MSE, MAE, RMSE, and R² metrics:

| Metric | Description |
|--------|-------------|
| `sparsity` | Fraction of zero entries in the target |
| `zero_precision` | Of entries predicted zero, fraction truly zero |
| `zero_recall` | Of truly zero entries, fraction predicted zero |
| `nonzero_precision` | Of entries predicted nonzero, fraction truly nonzero |
| `nonzero_recall` | Of truly nonzero entries, fraction predicted nonzero |

These appear automatically in `test_report.json` and cross-validation outputs.

## Mathematical documentation

The [`docs/`](docs/) directory contains detailed theory write-ups for every model
architecture, each available in both Markdown and LaTeX:

| Document | Model |
|----------|-------|
| [`vae_theory`](docs/vae_theory.md) | β-VAE / Vanilla VAE |
| [`hyperbolicvae_theory`](docs/hyperbolicvae_theory.md) | Hyperbolic VAE (Poincaré ball) |
| [`graphvae_theory`](docs/graphvae_theory.md) | Graph-regularized taxonomy VAE |
| [`treepriorvae_theory`](docs/treepriorvae_theory.md) | Tree-prior VAE (Brownian motion) |
| [`phylofusionvae_theory`](docs/phylofusionvae_theory.md) | Phylogenetic fusion VAE |
| [`tree_dtm_vae_theory`](docs/tree_dtm_vae_theory.md) | TreeDTM-VAE (tree-multinomial / Dirichlet-tree-multinomial / Dirichlet-tree) |
| [`philrvae_theory`](docs/philrvae_theory.md) | PhILR-VAE (isometric log-ratio; five compositional likelihoods) |
| [`hyperbolic_philrvae_theory`](docs/hyperbolic_philrvae_theory.md) | Hyperbolic PhILR-VAE (Poincaré ball + PhILR; logmap0 audit fix) |
| [`diva_theory`](docs/diva_theory.md) | DIVA — domain-invariant latent partition `z = [z_d ; z_y ; z_x]` |
| [`phylodiva_theory`](docs/phylodiva_theory.md) | PhyloDIVA — DIVA + GRL critic + Deep CORAL + phylogenetic smoothness |
| [`flowxformervae_theory`](docs/flowxformervae_theory.md) | FlowXFormerVAE *(deprecated)* |
| [`hgvae_zi_theory`](docs/hgvae_zi_theory.md) | HG-VAE-ZI *(deprecated)* |

Each document formalises the loss function, encoder/decoder architecture, and
any model-specific priors or transforms. The LaTeX versions (`.tex`) are
suitable for inclusion in manuscripts. See
[`docs/theory_coverage.md`](docs/theory_coverage.md) for a checklist of
topics covered per model.
