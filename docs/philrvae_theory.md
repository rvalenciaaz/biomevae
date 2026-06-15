# PhILR-VAE Theory

This document formalises the **PhILRVAE** model implemented in
`src/biomevae/models/philrvae.py`. The current implementation is built on
the compositional `TaxonomyGraph` backbone (`taxonomy_tree.py`) and supports
**five** reconstruction likelihoods chosen via the `likelihood` argument.

The earlier "PhILR + leaf-NB" variant has been retired: an independent
Negative Binomial on each leaf is incoherent for closed compositional data
and double-models zeros that the simplex / Dirichlet handles natively.

---

## 1. Problem setting

Let

- \(\mathbf x \in \mathbb N_0^p\) be observed counts for \(p\) taxa/features,
  or \(\mathbf x \in \Delta^{p-1}\) for relative-abundance data,
- \(N = \sum_i x_i\) the library size (for count input),
- \(\Delta^{p-1} = \{\mathbf c \in \mathbb R_{>0}^p : \sum_i c_i = 1\}\) the
  open simplex,
- \(\mathbf z \in \mathbb R^d\) the latent.

The model separates **composition** (relative abundances, via PhILR / ILR
coordinates) from **depth/scale** (library size \(N\), reintroduced
analytically only by the count likelihoods).

Given a small pseudocount \(\alpha > 0\) (count input) or \(\alpha \to 0^+\)
(relative input — `relative_pseudocount`), the closed composition is

\[
c_i = \frac{x_i + \alpha}{\sum_{j} (x_j + \alpha)}.
\]

The TaxonomyGraph backbone (`build_philr_basis_from_taxonomy_graph`)
provides the rooted tree, leaf indexing, and sibling-group bookkeeping
shared with TreeDTM-VAE.

---

## 2. PhILR transform as an isometry

### 2.1 Contrast basis from Sequential Binary Partition (SBP)

A rooted taxonomy induces a Sequential Binary Partition. At every internal
node with two disjoint descendant-leaf groups \(R, S\) of cardinalities
\(r, s\), the corresponding balance vector \(\boldsymbol\psi \in \mathbb R^p\)
has entries

\[
\psi_i =
\begin{cases}
+\sqrt{\dfrac{s}{r(r+s)}} & i \in R,\\
-\sqrt{\dfrac{r}{s(r+s)}} & i \in S,\\
0 & \text{otherwise.}
\end{cases}
\]

Internal nodes with \(K > 2\) children are linearised into \(K-1\) ordered
splits (left-vs-rest, then left-vs-rest on the remainder, …) — see the loop
in `build_philr_basis_from_taxonomy_graph`. This produces

- \(\langle \boldsymbol\psi, \mathbf 1\rangle = 0\) (contrast),
- \(\|\boldsymbol\psi\|_2 = 1\),
- mutual orthogonality across all valid SBP contrasts.

Stacking \(p-1\) balances yields the matrix
\(\Psi \in \mathbb R^{p \times (p-1)}\) with orthonormal columns spanning the
clr-hyperplane \(\mathbf 1^\perp\). Orthonormality is verified at
construction (`check_orthonormal=True`), with a hard error if \(\Psi^\top\Psi\)
deviates from \(I_{p-1}\).

### 2.2 Forward / inverse

PhILR coordinates and their inverse:

\[
\mathbf y = \mathrm{ilr}(\mathbf c) = \log(\mathbf c)\,\Psi \in \mathbb R^{p-1},
\qquad
\hat{\mathbf c} = \mathrm{softmax}(\hat{\mathbf y} \Psi^\top) \in \Delta^{p-1}.
\]

The softmax restores the closure dropped by ilr; the pair is exact on the
clr-hyperplane.

### 2.3 Metric equivalence

For any two compositions
\(\mathbf c^{(1)}, \mathbf c^{(2)} \in \Delta^{p-1}\) with PhILR images
\(\mathbf y^{(k)}\),

\[
\|\mathbf y^{(1)} - \mathbf y^{(2)}\|_2 = d_A(\mathbf c^{(1)}, \mathbf c^{(2)}),
\]

where \(d_A\) is the Aitchison distance. Euclidean operations in PhILR space
are therefore geometrically correct for compositional data.

---

## 3. Encoder / decoder parameterisation

\[
p(\mathbf z) = \mathcal N(\mathbf 0, I_d),
\qquad
q_\phi(\mathbf z \mid \mathbf x) = \mathcal N\!\big(\boldsymbol\mu_\phi(\mathbf x),\,\mathrm{diag}\,\boldsymbol\sigma^2_\phi(\mathbf x)\big).
\]

Encoder pipeline:

\[
\mathbf x \to \mathbf c \to \mathbf y = \mathrm{ilr}(\mathbf c) \to h_\phi(\mathbf y) \to (\boldsymbol\mu_\phi, \log \boldsymbol\sigma_\phi^2).
\]

Reparameterised sampling:
\(\mathbf z = \boldsymbol\mu_\phi + \boldsymbol\sigma_\phi \odot \boldsymbol\varepsilon\).

Decoder pipeline:

\[
\mathbf z \to g_\theta(\mathbf z) = \hat{\mathbf y} \in \mathbb R^{p-1}
\to \hat{\mathbf c} = \mathrm{softmax}(\hat{\mathbf y}\Psi^\top) \in \Delta^{p-1}.
\]

For tree-aware likelihoods (`dirichlet_tree_multinomial`, `dirichlet_tree`)
the decoder additionally produces a per-sibling-group concentration
\(\alpha^0_g > 0\) (same `TreeDTMDecoder` head used by `TreeDTMVAE`).

---

## 4. Reconstruction likelihoods

`PhILRVAE` selects one of five forms via `likelihood`. All five share the
encoder / decoder and KL term; only the log-density of \(\mathbf x\) given
\((\hat{\mathbf c}, \hat{\mathbf y})\) changes.

### 4.1 `philr_gaussian` (default for relative-abundance input)

A diagonal Gaussian on ILR coordinates:

\[
\mathbf y = \mathrm{ilr}(\mathbf c)
\sim \mathcal N(\hat{\mathbf y}, \mathrm{diag}\,\boldsymbol\sigma_\text{obs}^2),
\]

with a learnable per-coordinate observation scale
(`init_coord_scale`, clamped to `min_coord_scale`). The negative log
density is the standard half-quadratic plus log-determinant term.

Because PhILR is an isometry, this Gaussian is the **logistic-normal** model
on \(\Delta^{p-1}\) — the maximum-entropy distribution on the simplex with
fixed Aitchison mean and covariance, and the canonical compositional
analogue of an MSE loss.

### 4.2 `multinomial` (integer counts, no overdispersion)

\[
\mathbf x \mid \hat{\mathbf c}, N \sim \mathrm{Multinomial}(N, \hat{\mathbf c}),
\quad
-\log p = -\log\binom{N}{\mathbf x} - \sum_i x_i \log \hat c_i.
\]

The library size \(N\) is reintroduced analytically; no separate scale
network is learned.

### 4.3 `dirichlet_multinomial` (overdispersed integer counts)

Replace the multinomial with a global Dirichlet prior of concentration
\(\alpha_0 > 0\) (single learnable scalar — `init_concentration`) and the
predicted composition as the mean direction:

\[
\boldsymbol\alpha = \alpha_0 \hat{\mathbf c},
\quad
-\log p(\mathbf x \mid \boldsymbol\alpha, N)
= -\log \binom{N}{\mathbf x}
- \log \Gamma(\alpha_0) + \log \Gamma(N + \alpha_0)
- \sum_i \!\Big(\log \Gamma(x_i + \alpha_i) - \log \Gamma(\alpha_i)\Big).
\]

Captures *global* count overdispersion. Use when counts are real but the
tree structure does not need to be reflected in the dispersion parameter.

### 4.4 `dirichlet_tree_multinomial` (tree-local overdispersion, integer counts)

Identical likelihood to **TreeDTM-VAE** §2.2 — see
[`tree_dtm_vae_theory.md`](tree_dtm_vae_theory.md): a product of
Dirichlet-Multinomial PMFs over internal nodes, with a **per sibling group**
concentration \(\alpha^0_g\) (rank-aware overdispersion).

The PhILR encoder is reused; the likelihood reads the same node-aggregated
counts \(\mathbf y\) that the TreeDTM family does, via
`aggregate_leaf_matrix_to_nodes(\hat{\mathbf c} \cdot N)`.

### 4.5 `dirichlet_tree` (continuous relative abundances)

For data already in \(\Delta^{p-1}\): product of continuous Dirichlet
densities at each internal node — see
[`tree_dtm_vae_theory.md`](tree_dtm_vae_theory.md) §2.3. Identical
implementation; the PhILR machinery only feeds the encoder.

### 4.6 Choosing a likelihood

| Input | Recommended `likelihood` |
|---|---|
| Closed relative abundances (\(\sum_i x_i = 1\)) | `philr_gaussian` or `dirichlet_tree` |
| Integer counts, low dispersion (rarefied / equal depth) | `multinomial` |
| Integer counts, global overdispersion | `dirichlet_multinomial` |
| Integer counts, clade-specific overdispersion | `dirichlet_tree_multinomial` |

---

## 5. ELBO objective

For each likelihood:

\[
\mathcal L
= -\mathbb E_{q_\phi}[\log p_\theta(\mathbf x \mid \mathbf z)]
+ \beta_t\,\mathrm{KL}\!\big(q_\phi(\mathbf z \mid \mathbf x) \,\|\, \mathcal N(\mathbf 0, I_d)\big)
+ \gamma\,\mathcal R(\boldsymbol\theta_\text{lik}),
\]

with closed-form Gaussian KL:

\[
\mathrm{KL} = \tfrac12 \sum_{j=1}^d \big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\big),
\]

and \(\beta_t\) following the linear warm-up schedule used everywhere in
`biomevae.trainers`. \(\mathcal R\) is a small L2 penalty on the relevant
likelihood-specific parameters
(\(\boldsymbol\sigma_\text{obs}\) for `philr_gaussian`, \(\alpha_0\) for
Dirichlet-Multinomial, \(\boldsymbol\alpha^0_g\) for tree variants).

The PhILR isometry means **no extra geometric / consistency penalty is
needed**: Euclidean decoding in ILR space and softmax-inverse closure are
already exact on the simplex.

---

## 6. Why this architecture is principled

1. **Compositional validity:** outputs are always in the simplex after
   inverse PhILR + softmax. The five likelihoods are all natural
   distributions on either \(\Delta^{p-1}\) (`philr_gaussian`,
   `dirichlet_tree`) or \(\mathbb N_0^p\) at fixed \(N\) (multinomial,
   Dirichlet-Multinomial, Dirichlet-Tree-Multinomial).
2. **Tree awareness:** the contrast basis is built from the taxonomy SBP,
   and tree-aware likelihoods reuse the same sibling-group decomposition as
   TreeDTM-VAE.
3. **Metric correctness:** Euclidean operations in PhILR space preserve
   Aitchison geometry exactly.
4. **Depth decoupling:** library size enters only through the count
   likelihoods, not the latent representation.
5. **No statistically incoherent NB:** the previous PhILR-NB / PhILR-ZINB
   variants are removed — those families fight the closure constraint
   rather than respecting it.

---

## 7. Computational graph

```mermaid
flowchart LR
    X["Raw input (counts or relative)"] --> C["Pseudocount + closure<br/>c in Delta^(p-1)"]
    C --> Y["PhILR: y = log(c) Psi"]
    Y --> ENC[Encoder MLP]
    ENC --> MU[mu(x)]
    ENC --> LV[logvar(x)]
    MU --> Z["z = mu + sigma * eps"]
    LV --> Z
    Z --> DEC[Decoder MLP]
    DEC --> YH[Predicted ILR y_hat]
    DEC --> A0["alpha0_g (tree variants)"]
    YH --> CH["c_hat = softmax(y_hat Psi^T)"]
    CH --> LIK["Likelihood:<br/>philr_gaussian | multinomial |<br/>dirichlet_multinomial |<br/>dirichlet_tree_multinomial |<br/>dirichlet_tree"]
    A0 --> LIK
    X --> LIK
    MU --> KL["KL(q || N(0,I))"]
    LV --> KL
    LIK --> L["Loss = NLL + beta * KL + gamma * R"]
    KL --> L
```

---

## 8. Implementation correspondence

| Math | Code |
|---|---|
| Closed composition \(\mathbf c\) | `close_composition()` in `taxonomy_tree.py` |
| SBP basis \(\Psi\) | `build_philr_basis_from_taxonomy_graph()` |
| Forward / inverse PhILR | `PhILRTransform.forward` / `.inverse` (in `philrvae.py`) |
| Encoder / decoder | `PhILRVAE.encode`, `reparam`, `decode`, `forward` |
| `philr_gaussian` NLL | `PhILRVAE.philr_gaussian_nll` |
| `multinomial` / `dirichlet_multinomial` NLL | `PhILRVAE.multinomial_nll` / `PhILRVAE.dirichlet_multinomial_nll` |
| Tree-likelihood NLLs | reused from `tree_dtm_vae.TreeDTMVAE` |
| KL + warm-up | `PhILRVAE.kl_per_sample`, `losses.beta_schedule` |
| CLI entry point | `biomevae.cli.vae_train_philrvae` |

---

## 9. Practical notes

- The orthonormality check `check_basis=True` is on by default; it is fast
  (one Gram-matrix multiply) and catches malformed taxonomy trees early.
- `sort_children=True` produces a canonical leaf order from the tree
  topology — required for reproducibility when several taxonomy files map to
  the same logical tree.
- All tree-aware likelihoods read the **same** node-aggregated tensor that
  TreeDTM-VAE uses; the two models can be trained on identical input data
  pipelines.
