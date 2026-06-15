# PhyloDIVA — phylogeny-aware domain adaptation on top of DIVA

This document describes the **PhyloDIVA** family of models, implemented as
domain-adaptive wrappers on the three DIVA backbones:

| Wrapper | Backbone | File |
|---|---|---|
| `PhyloDIVABetaVAE` | β-VAE | `models/phylodiva_betavae.py` |
| `PhyloDIVAHyperbolicPhILRVAE` | Hyperbolic PhILR | `models/phylodiva_hyp_philrvae.py` |
| `PhyloDIVATreeDTMVAE` | TreeDTM | `models/phylodiva_treedtmvae.py` |

PhyloDIVA inherits everything from DIVA — latent partition, conditional
priors, auxiliary classifiers — see [`diva_theory.md`](diva_theory.md) — and
adds **three** regularisers that close the gaps left by vanilla DIVA on
leave-one-study-out (LOSO) microbiome data:

1. a **gradient-reversed latent study critic** on `z_y`,
2. **Deep CORAL** covariance matching on `z_x` per study,
3. **Brownian-motion / phylogenetic smoothness** on the decoder's
   tree-edge or PhILR-contrast outputs.

The supporting machinery lives in
`src/biomevae/models/phylo_da.py` (GRL critic + CORAL) and
`src/biomevae/models/phylo_cov.py` (smoothness penalties).

---

## 1. Why PhyloDIVA exists

Vanilla DIVA enforces "predict domain from \(z_d\)" through the auxiliary
cross-entropy `CE_d`. It does **not** enforce "scrub study from \(z_y\)".
In LOSO settings this leaks: a classifier trained on `z_y` can still
distinguish studies, which means held-out studies are out of distribution
for the disease classifier. The three regularisers below close this gap
while keeping DIVA's likelihood-agnostic structure intact.

## 2. Adversarial study critic on \(z_y\)

A small MLP `LatentStudyCritic` predicts the study id from \(z_y\) **through
a gradient reversal layer (GRL)**:

\[
\mathbf z_y
\xrightarrow{\;\text{GRL}(\lambda_t)\;} \mathbf z_y'
\xrightarrow{\;\text{MLP}\;} \hat{\mathbf p}(d \mid \mathbf z_y),
\]

\[
\mathcal L_\text{GRL}^{(i)} = -\log \hat p\!\big(d^{(i)} \,\big|\, \mathbf z_y^{(i)}\big).
\]

The GRL is the identity on the forward pass and multiplies the gradient
by \(-\lambda_t\) on the backward pass (Ganin & Lempitsky, ICML 2015). The
critic minimises cross-entropy while the **encoder maximises** it — `z_y`
is therefore driven to be un-predictive of study identity.

\(\lambda_t\) follows the canonical DANN sigmoid ramp
(`dann_lambda_schedule`):

\[
\lambda(t) = \lambda_\text{max}\!\left( \frac{2}{1 + e^{-\gamma t}} - 1 \right),
\qquad t \in [0, 1],\; \gamma = 10\text{ (default)},
\]

so the adversary is *off* at the start of training (\(\lambda(0)=0\)) and
approaches \(\lambda_\text{max}\) as training finishes. This avoids the
common DANN failure mode where a strong adversary collapses the encoder
before the reconstruction term has had a chance to organise the latent.

### 2.1 Why latent-space, not input-space

An earlier draft used a hierarchical critic on internal-node abundances
aggregated from the leaf input. That critic was a no-op on the encoder:
the aggregator has no learnable parameters, so the GRL gradient flowed
back to a dataloader leaf with `requires_grad=False` and stopped there.
A latent-space critic on \(\mathbf z_y\) routes the GRL gradient through
the entire encoder, which is the correct adversarial target.

## 3. Deep CORAL on \(\mathbf z_x\)

For each study \(k\) present in the batch with \(n_k \ge 2\) samples,
estimate the per-study covariance of the residual latent factor:

\[
\hat\Sigma_k = \frac{1}{n_k - 1} \sum_{i:\,d^{(i)}=k} (\mathbf z_x^{(i)} - \bar{\mathbf z}_{x,k})(\mathbf z_x^{(i)} - \bar{\mathbf z}_{x,k})^\top.
\]

The CORAL loss (Sun & Saenko 2016) is the average pairwise Frobenius
distance between per-study covariances, normalised by latent dimension and
the number of valid pairs:

\[
\mathcal L_\text{CORAL}
= \frac{1}{|P|}\sum_{(k,k') \in P}\!
\frac{1}{4 p^2}\,
\big\|\hat\Sigma_k - \hat\Sigma_{k'}\big\|_F^{\,2},
\]

with \(P = \{(k,k') : k < k',\; n_k, n_{k'} \ge m\}\) (default
`min_per_study = 2`) and \(p = \dim(\mathbf z_x)\). The implementation
(`coral_per_study`) is fully vectorised via a one-hot membership matrix
and `einsum` so the loss is **deterministic** under
`torch.use_deterministic_algorithms` when `CUBLAS_WORKSPACE_CONFIG` is set.

Intuition: a domain-invariant residual representation should look like
the *same* second-order distribution across studies, regardless of which
study contributed the samples.

## 4. Phylogenetic smoothness on the decoder

The decoder outputs are tied to the tree structure of the input — edge
logits for `TreeDTMVAE`, PhILR contrasts for the PhILR backbones. The
smoothness term is a discrete analogue of a Brownian-motion prior over a
Gaussian process indexed by tree position (Felsenstein 1985): adjacent
edges/contrasts should differ by an amount proportional to the branch
length between them.

### 4.1 Edge-tensor smoothness (TreeDTM backbone)

For each edge \(e\) with parent edge \(\mathrm{pa}(e)\) and branch length
\(\ell_e > 0\),

\[
\mathcal L_\text{BM-edge}
= \frac{1}{|E^\star|} \sum_{e \in E^\star}\!
\frac{(a_e - a_{\mathrm{pa}(e)})^2}{\ell_e + \varepsilon},
\]

where \(a_e\) is the decoder logit on edge \(e\) and \(E^\star\) is the
set of non-root edges. The parent-edge index is precomputed with
`build_edge_parent_edge_index`; the penalty is `bm_edge_smoothness`.

### 4.2 PhILR-coordinate smoothness (PhILR backbones)

PhILR contrasts are gauge-fixed (sum-to-zero within each split), so the
analogous penalty is on adjacent contrasts in the post-order traversal of
the SBP tree:

\[
\mathcal L_\text{BM-coord}
= \frac{1}{|C^\star|} \sum_{c \in C^\star}\!
\frac{(\hat y_c - \hat y_{\mathrm{pa}(c)})^2}{\ell_c + \varepsilon}.
\]

The implementation (`bm_coord_smoothness`) walks the same post-order used
by `build_philr_basis_from_taxonomy_graph` so the contrast indices match.

### 4.3 Caveats

- The "BM prior" interpretation only holds **when branch lengths are
  calibrated** (e.g. ultrametric or expected substitutions). For
  uncalibrated taxonomies, the term is best read as a generic
  regularisation prior favouring smoothness along the tree, not a literal
  Brownian motion likelihood. Passing `edge_length=None` falls back to
  unweighted differences, which is the safer default for purely
  topological taxonomies.

## 5. Composite objective

The PhyloDIVA loss adds three terms to the DIVA loss of
[`diva_theory.md`](diva_theory.md) §4:

\[
\boxed{\,
\mathcal L_\text{PhyloDIVA}
= \mathcal L_\text{DIVA}
+ \lambda_t \,\mathcal L_\text{GRL}
+ \mu \,\mathcal L_\text{CORAL}
+ \nu \,\mathcal L_\text{BM},
\,}
\]

with

- \(\lambda_t\) — DANN sigmoid ramp on the GRL critic (§2);
- \(\mu\) — CORAL weight (CLI flag `--coral-weight`);
- \(\nu\) — smoothness weight (CLI flag `--smooth-weight`).

Reconstruction NLL, three KL terms, and auxiliary classifier CEs are
inherited from the DIVA backbone wrapper unchanged.

## 6. Diagnostic expectations

For a held-out study a correctly trained PhyloDIVA model should give:

| Quantity | DIVA only | PhyloDIVA |
|---|---|---|
| Domain accuracy of `q(d \| z_d)` | high | high |
| Domain accuracy of a fresh classifier on `z_y` | high (leakage) | **near chance** |
| Pair-wise Frobenius distance of `cov(z_x \| study)` | non-zero | shrinks during training |
| Class accuracy of `q(y \| z_y)` on held-out study | drops | preserved |

The `loso_summary_*.tsv` artefacts emitted by the training pipeline
(`results/`) track exactly these metrics across the five canonical seeds
documented in the top-level `README.md`.

## 7. Implementation correspondence

| Math | Code |
|---|---|
| Gradient reversal layer | `GradientReversal` in `models/grl.py` |
| Latent study critic | `LatentStudyCritic` in `models/phylo_da.py` |
| DANN sigmoid schedule \(\lambda(t)\) | `dann_lambda_schedule` |
| Deep CORAL \(\mathcal L_\text{CORAL}\) | `coral_per_study` |
| Hierarchical aggregators (for input-space diagnostics) | `build_internal_aggregator` in `models/phylo_cov.py` |
| Edge-parent index | `build_edge_parent_edge_index` |
| Edge-BM smoothness | `bm_edge_smoothness` |
| Internal-node-parent index for PhILR | `build_internal_node_parent_idx` |
| Coord-BM smoothness for PhILR | `bm_coord_smoothness` |
| Per-backbone wrappers | `models/phylodiva_{betavae,hyp_philrvae,treedtmvae}.py` |
| CLI entry points | `biomevae-train-phylodiva-betavae`, `biomevae-train-phylodiva-hyp-philrvae`, `biomevae-train-phylodiva-tree-dtm` |

## 8. References

- Ganin & Lempitsky, *Unsupervised Domain Adaptation by Backpropagation*,
  ICML 2015 (gradient-reversal layer + DANN schedule).
- Sun & Saenko, *Deep CORAL: Correlation Alignment for Deep Domain
  Adaptation*, ECCV 2016.
- Felsenstein, *Phylogenies and the comparative method*, American
  Naturalist 1985 (Brownian-motion prior on trees).
- Ilse, Tomczak, Louizos, Welling, *DIVA: Domain Invariant Variational
  Autoencoders*, MIDL 2020.
