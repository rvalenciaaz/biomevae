# Evaluation, reconstruction, and classification

[← back to main README](README.md)

This page covers the post-training pipeline: extracting embeddings,
interpreting them against the original OTUs, running held-out
reconstruction tests, comparing trained models to the NMF baseline
using Gabriel bi-cross-validation, and classifying metadata labels
from the learned latent spaces. Every evaluation CLI on this page runs
across the canonical 5-seed protocol documented in the
[Reproducibility and seeds](README.md#reproducibility-and-seeds) section
of the main README.

- [Embedding extraction](#embedding-extraction-all-model-types)
- [Embedding interpretation](#embedding-interpretation)
- [Model testing](#model-testing-all-model-types)
- [NMF baseline cross-validation](#nmf-baseline-cross-validation)
- [Comparing neural models to NMF](#comparing-neural-models-to-nmf)
- [Metadata classification from embeddings](#metadata-classification-from-embeddings)

## Embedding extraction (all model types)

```bash
biomevae-embed   --input sgb_table.tsv   --model-dir runs/hyp_tax   --outdir runs/hyp_tax/embed_eval   --emb-space ball --export-recon   --taxonomy phyla.tsv
```

`biomevae-embed` reloads the saved `config.json` and `model.pt`, applies the stored
preprocessing pipeline, and writes `embeddings.tsv` in the requested latent
space. Supplying `--export-recon` saves `recon.tsv` alongside the embeddings so
that downstream ordination or visualization tools can operate directly on the
outputs.

### Multi-model embedding extraction

Run embedding extraction for each model type with the same flags used during training:

```bash
biomevae-embed --input sgb_table.tsv --model-dir runs/base --outdir runs/base/embed
biomevae-embed --input sgb_table.tsv --model-dir runs/vanilla --outdir runs/vanilla/embed
biomevae-embed --input sgb_table.tsv --model-dir runs/tax --outdir runs/tax/embed --taxonomy phyla.tsv
biomevae-embed --input sgb_table.tsv --model-dir runs/train-graph --outdir runs/train-graph/embed --taxonomy phyla.tsv
biomevae-embed --input sgb_table.tsv --model-dir runs/tree-prior --outdir runs/tree-prior/embed --taxonomy phyla.tsv
biomevae-embed --input sgb_table.tsv --model-dir runs/tree-dtm-vae --outdir runs/tree-dtm-vae/embed --taxonomy phyla.tsv
biomevae-embed --input sgb_table.tsv --model-dir runs/philrvae --outdir runs/philrvae/embed --taxonomy phyla.tsv
biomevae-embed --input sgb_table.tsv --model-dir runs/train-fuse --outdir runs/train-fuse/embed --taxonomy phyla.tsv
biomevae-embed --input sgb_table.tsv --model-dir runs/hyp --outdir runs/hyp/embed --emb-space ball
biomevae-embed --input sgb_table.tsv --model-dir runs/hyp_tax --outdir runs/hyp_tax/embed --emb-space ball --taxonomy phyla.tsv
```

## Embedding interpretation

```bash
biomevae-interpret \
    --input sgb_table.tsv \
    --model-dir runs/hyp_tax \
    --outdir runs/hyp_tax/interpret \
    --background-size 128 \
    --explain-size 256 \
    --top-k 20
```

The interpreter links the latent representation back to the original OTU
abundances using SHAP (DeepExplainer), mutual information, and Spearman
correlation analyses. It writes latent embeddings, SHAP summaries, and
feature-ranking tables (`otu_latent_summary.tsv`) that identify the taxa with
the strongest influence on each latent dimension. Install the optional
dependency group with `pip install -e .[interpret]` to enable the SHAP-based
analysis. Pass `--save-sample-shap` to export the full SHAP tensor for
downstream visualization or custom aggregation. When SHAP warns that the
background is large, add `--background-summary sample` (or `kmeans`) to replace
the raw subset with a smaller representative set; control the final size with
`--background-summary-size`. Use `--shap-nsamples` to cap the KernelExplainer
evaluations (lower values run faster with less precise attributions). For
tables with a few hundred samples, the `--background-size 128` /
`--explain-size 256` defaults above stay below the threshold that triggered SHAP
warnings in our runs; scale them down further if your command still reports
slowdowns.

### Multi-model embedding interpretation

Run embedding interpretation for each model type with the same flags used during training:

```bash
biomevae-interpret --input sgb_table.tsv --model-dir runs/base --outdir runs/base/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/vanilla --outdir runs/vanilla/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/tax --outdir runs/tax/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/train-graph --outdir runs/train-graph/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/tree-prior --outdir runs/tree-prior/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/train-fuse --outdir runs/train-fuse/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/hyp --outdir runs/hyp/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/hyp_tax --outdir runs/hyp_tax/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/tree-dtm-vae --outdir runs/tree-dtm-vae/interpret
biomevae-interpret --input sgb_table.tsv --model-dir runs/philrvae --outdir runs/philrvae/interpret
```

To aggregate feature interpretation at a taxonomy level (e.g., species, genus, family), supply the taxonomy table and a level:

```bash
biomevae-interpret \
    --input sgb_table.tsv \
    --model-dir runs/hyp_tax \
    --outdir runs/hyp_tax/interpret_genus \
    --taxonomy phyla.tsv \
    --taxonomy-level genus
```

Run genus-level interpretation for each model type:

```bash
biomevae-interpret --input sgb_table.tsv --model-dir runs/base --outdir runs/base/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/vanilla --outdir runs/vanilla/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/tax --outdir runs/tax/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/train-graph --outdir runs/train-graph/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/tree-prior --outdir runs/tree-prior/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/train-fuse --outdir runs/train-fuse/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/hyp --outdir runs/hyp/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/hyp_tax --outdir runs/hyp_tax/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/tree-dtm-vae --outdir runs/tree-dtm-vae/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
biomevae-interpret --input sgb_table.tsv --model-dir runs/philrvae --outdir runs/philrvae/interpret_genus --taxonomy phyla.tsv --taxonomy-level genus
```

## Model testing (all model types)

```bash
biomevae-test   --input heldout.tsv   --model-dir runs/hyp_tax   --outdir runs/hyp_tax/test_on_heldout   --export   --taxonomy phyla.tsv
biomevae-test   --input heldout.tsv   --model-dir runs/tree-dtm-vae --outdir runs/tree-dtm-vae/test_on_heldout --export --taxonomy phyla.tsv
```

`biomevae-test` mirrors the training-time preprocessing before computing the
reconstruction and KL terms configured for the model. Metrics are exported to
`test_report.json`; adding `--export` also writes `embeddings.tsv` and
`recon.tsv` in the output directory for inspection or reuse.

## NMF baseline cross-validation

Use the `biomevae-nmf` entry point to evaluate the classical NMF baseline using Gabriel-style holdouts for a bi-cross-validation scheme. Supply the
counts table, the desired number of components, and (optionally) additional
keyword arguments forwarded to `sklearn.decomposition.NMF`. To pick the rank
automatically, pass `--rank-candidates` with a comma-separated list or a range:

```bash
biomevae-nmf   --input sgb_table.tsv   --components 16   --splits 5   --train-fraction 0.9 \
             --nmf-kw init=nndsvda --nmf-kw max_iter=500   --output runs/nmf_summary.json

biomevae-nmf   --input sgb_table.tsv   --rank-candidates 8,12,16,24   --splits 5 \
             --train-fraction 0.9   --output runs/nmf_rank_selected.json
```

The CLI prints the reconstruction summary to stdout and writes it to `--output`
if provided. Disable the default `log1p` preprocessing with `--no-log1p`.

### Gabriel-style bi-cross-validation for VAEs and NMF

The helper `gabriel_split` implements the Owen–Perry (2009) Gabriel bi-cross-validation procedure for a count matrix
\(X \in \mathbb{R}_{\ge 0}^{n \times p}\). For a requested training fraction \(\rho\), the routine draws
\(r = \lfloor \sqrt{(1-\rho) n} \rfloor\) validation rows and \(c = \lfloor \sqrt{(1-\rho) p} \rfloor\) validation columns after shuffling
the indices. The held-out block is therefore the Cartesian product
\(
\mathcal{H} = R_{\text{val}} \times C_{\text{val}},\quad |R_{\text{val}}| = r,\; |C_{\text{val}}| = c,
\)
and the training data are the remaining rows:
\[
X_{\text{train}} = X_{R_{\text{train}}, :},\quad R_{\text{train}} = [n] \setminus R_{\text{val}}.
\]

Every fold trains its model using only \(X_{\text{train}}\); none of the entries in \(\mathcal{H}\) influence the fit. At inference time the
column holdout is enforced for both model families so that \(C_{\text{val}}\) features cannot inform the latent representation:

- **VAE:** the held-out columns of each validation row are zeroed out before
  encoding, so the encoder derives its latent code only from
  \(C_{\text{train}}\) features.
- **NMF:** validation-row loadings \(W_{\text{val}}\) are obtained by solving a
  non-negative least-squares projection against the components restricted to
  \(C_{\text{train}}\), i.e.\
  \(\min_{w \ge 0} \| X_{i, C_{\text{train}}} - w\, H_{:, C_{\text{train}}} \|_2\)
  for each validation sample \(i\).

In both cases the full decoder / components matrix is then used to produce
reconstructions \(\hat{X}\) across all features, and metrics are reported on
the held-out block:
\[
\text{MSE} = \frac{1}{|\mathcal{H}|} \sum_{(i,j) \in \mathcal{H}} \bigl( X_{ij} - \hat{X}_{ij} \bigr)^2,
\quad
\text{MAE} = \frac{1}{|\mathcal{H}|} \sum_{(i,j) \in \mathcal{H}} \bigl| X_{ij} - \hat{X}_{ij} \bigr|,
\]
with analogous formulas for RMSE and \(R^2\). This mirrors the canonical Gabriel protocol: the column holdout is applied during both inference
and evaluation, ensuring that each model must predict the withheld entries without ever observing them as input context.

## Comparing neural models to NMF

Two helper commands run side-by-side evaluations of trained VAEs against the
NMF baseline using the same cross-validation splits.

### Single method comparison

```bash
biomevae-comparetonmf   --input sgb_table.tsv   --method-name hyp_tax \
                     --method-config runs/hyp_tax/config.json \
                     --components 16   --splits 5   --train-fraction 0.9 \
                     --taxonomy phyla.tsv \
                     --output runs/hyp_tax_vs_nmf.json

biomevae-comparetonmf   --input sgb_table.tsv   --method-name hyp_tax \
                     --method-config runs/hyp_tax/config.json \
                     --rank-candidates 8-24   --splits 5   --train-fraction 0.9 \
                     --taxonomy phyla.tsv \
                     --output runs/hyp_tax_vs_nmf.json
```

`--method-config` should point to the JSON produced during training (for Optuna
studies, use the best-trial configuration). Override the device for evaluation
with `--device cpu` or `--device cuda` as needed. Providing `--taxonomy`
enables the same hierarchy-aware metrics that `biomevae-allcomp` reports for all
evaluated models, including the NMF baseline.

The resulting JSON report contains the VAE metrics, NMF baseline scores, and the
paths to the artifacts used for comparison, making it straightforward to cite in
the accompanying manuscript.

### Multiple methods at once

```bash
biomevae-allcomp   --input sgb_table.tsv   --components 16   --splits 5 \
                 --method base=runs/base/config.json \
                 --method vanilla=runs/vanilla/config.json \
                 --method tax=runs/tax/config.json \
                 --method graph=runs/train-graph/config.json \
                 --method treeprior=runs/tree-prior/config.json \
                 --method tree-dtm-vae=runs/tree-dtm-vae/config.json \
                 --method philrvae=runs/philrvae/config.json \
                 --method fuse=runs/train-fuse/config.json \
                 --method hyp=runs/hyp/config.json \
                 --method hyp_tax=runs/hyp_tax/config.json \
                 --taxonomy phyla.tsv \
                 --output runs/all_methods_vs_nmf.json

biomevae-allcomp   --input sgb_table.tsv   --rank-candidates 8-24   --splits 5 \
                 --method base=runs/base/config.json \
                 --method vanilla=runs/vanilla/config.json \
                 --method tax=runs/tax/config.json \
                 --method graph=runs/train-graph/config.json \
                 --method treeprior=runs/tree-prior/config.json \
                 --method tree-dtm-vae=runs/tree-dtm-vae/config.json \
                 --method philrvae=runs/philrvae/config.json \
                 --method fuse=runs/train-fuse/config.json \
                 --method hyp=runs/hyp/config.json \
                 --method hyp_tax=runs/hyp_tax/config.json \
                 --taxonomy phyla.tsv \
                 --output runs/all_methods_vs_nmf.json
```

Repeat `--method NAME=PATH` for each trained model you want to include. Passing
`--device` forces all neural evaluations onto the specified backend; otherwise
each configuration’s `device` field is respected or defaulted to CPU.

`biomevae-allcomp` aggregates the outcomes from each configuration into a single
summary JSON that mirrors the schema written by the single-method comparison,
facilitating table generation inside `docs/paper/main.tex`. When `--taxonomy`
is provided the tool also computes hierarchy-aware metrics (MAE/RMSE/R²) on the
aggregated validation counts for every model in the comparison—including
methods that were trained without taxonomy losses—highlighting the latent-space
advantages of the models that optimise the hierarchy losses.

## Metadata classification from embeddings

Use `biomevae-classify` to evaluate how well the learned VAE embeddings
separate sample groups (e.g. CRC vs healthy).  Four classifiers are
evaluated using repeated stratified k-fold cross-validation:

```bash
biomevae-classify \
    --embeddings runs/philrvae/embed/embeddings.tsv \
    --metadata data/lij/sample_metadata.tsv \
    --label disease \
    --outdir runs/philrvae/classify \
    --n-splits 5 --n-repeats 10
```

Classifiers: Logistic Regression, Random Forest, SVM (RBF kernel),
Gradient Boosting.  Metrics: balanced accuracy, F1 (macro/weighted),
AUROC, confusion matrix, and full classification report.
