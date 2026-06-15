# Hyperbolic PhILR-VAE — Theory

This document describes **HyperbolicPhILRVAE** implemented in
`src/biomevae/models/hyperbolic_philrvae.py`. The model places the
[PhILR-VAE](philrvae_theory.md) family on a Poincaré-ball latent space; the
five compositional likelihoods are inherited unchanged.

The previous count-likelihood variants `HyperbolicPhILRNBVAE` and
`HyperbolicPhILRZINBVAE` have been **removed**. NB on relative-abundance
leaves is the wrong density, and ZINB double-models zeros that the simplex
and Dirichlet handle natively; only the compositional likelihoods documented
in [`philrvae_theory.md`](philrvae_theory.md) are exposed.

## 1. Setup

Let \(\mathbb B_c^d\) be the Poincaré ball of curvature \(-c\) (with
\(c > 0\)) and \(T_0 \mathbb B_c^d \cong \mathbb R^d\) its tangent space at
the origin. The encoder produces a tangent-space Gaussian whose draws are
pushed to the ball via the exponential map at the origin.

## 2. Generative model

\[
\mathbf v \sim q_\phi(\mathbf v \mid \mathbf x),
\qquad
\mathbf z = \mathrm{expmap}_0(\mathbf v) \in \mathbb B_c^d,
\]

\[
\hat{\mathbf y} = g_\theta\big(\mathrm{logmap}_0(\mathbf z)\big) \in \mathbb R^{p-1},
\qquad
\hat{\mathbf c} = \mathrm{softmax}(\hat{\mathbf y} \Psi^\top) \in \Delta^{p-1}.
\]

The reconstruction likelihood is one of the five exposed by `PhILRVAE` —
`philr_gaussian` (default), `multinomial`, `dirichlet_multinomial`,
`dirichlet_tree_multinomial`, or `dirichlet_tree`.

### 2.1 The `logmap0`-before-Linear refinement

The previous generation passed the ball point \(\mathbf z\) directly into a
Euclidean MLP, which mixes a hyperbolic point with linear operations that are
**not** isometries of the ball. The fix (commit `373a90f`, audit point D2)
is to lift \(\mathbf z\) back to the tangent space at the origin **before**
the first Linear layer of the decoder:

\[
\mathrm{logmap}_0(\mathbf z) = \frac{\mathrm{artanh}(\sqrt c \|\mathbf z\|)}{\sqrt c \|\mathbf z\|}\, \mathbf z.
\]

This makes the decoder operate on a Euclidean tangent vector with the
correct metric scaling, while still letting the latent representation live
in the ball.

## 3. Hyperbolic posterior

Tangent-space Gaussian parameters from the encoder:

\[
(\boldsymbol\mu_\text{tan}, \log \boldsymbol\sigma^2_\text{tan})
= f_\phi(\mathrm{ilr}(\mathbf x)).
\]

Reparameterised sampling (`HyperbolicPhILRVAE.reparam_to_ball`):

\[
\mathbf v = \boldsymbol\mu_\text{tan} + \boldsymbol\sigma_\text{tan} \odot \boldsymbol\varepsilon,
\quad \boldsymbol\varepsilon \sim \mathcal N(\mathbf 0, I_d),
\quad
\mathbf z = \mathrm{proj}_{\mathbb B}\big(\mathrm{expmap}_0(\mathbf v)\big).
\]

`expmap_0(v) = tanh(\sqrt c \|v\|)\,\mathbf v / (\sqrt c \|v\|)`, and
\(\mathrm{proj}_{\mathbb B}\) is the numerical projection back inside the
ball when floating-point drift puts \(\mathbf z\) past \(1/\sqrt c\). All
manifold ops are delegated to `geoopt.manifolds.PoincareBallExact`.

## 4. Prior and KL

We use the tangent-space Gaussian prior \(\mathcal N(\mathbf 0, I_d)\) on
\(\mathbf v\) and compute the KL in the tangent space:

\[
\mathrm{KL}\!\big(q_\phi(\mathbf v \mid \mathbf x) \,\|\, \mathcal N(\mathbf 0, I_d)\big)
= \tfrac12 \sum_j \big(\mu_{\text{tan},j}^2 + \sigma_{\text{tan},j}^2 - 1 - \log \sigma_{\text{tan},j}^2\big).
\]

Because \(\mathbf z\) is a deterministic function of \(\mathbf v\) through
\(\mathrm{expmap}_0\), the change-of-variable Jacobians of \(q\) and the
prior cancel and the manifold KL equals the tangent-space KL (see
[`hyperbolicvae_theory.md`](hyperbolicvae_theory.md) §6).

## 5. Reconstruction likelihoods

Identical to [`philrvae_theory.md`](philrvae_theory.md) §4 — pick the
appropriate density for the input type:

- `philr_gaussian` — logistic-normal on the simplex (default, relative input);
- `multinomial` — fixed-depth integer counts, no overdispersion;
- `dirichlet_multinomial` — globally overdispersed counts;
- `dirichlet_tree_multinomial` — clade-overdispersed counts (see [`tree_dtm_vae_theory.md`](tree_dtm_vae_theory.md));
- `dirichlet_tree` — continuous Dirichlet-tree on closed compositions.

No NB / ZINB head is exposed.

## 6. Objective

\[
\mathcal L = -\mathbb E_q[\log p_\theta(\mathbf x \mid \mathbf z)] + \beta_t \, \mathrm{KL} + \gamma \, \mathcal R,
\]

with the same `beta_schedule` warm-up and the same likelihood-specific L2
penalty \(\mathcal R\) as `PhILRVAE`.

## 7. Why this combination

PhILR gives analytic compositionality (Euclidean distance in ILR space =
Aitchison distance on the simplex). A Poincaré-ball latent gives
hierarchical geometry that matches the taxonomic tree underpinning the ILR
contrasts: the tree has exponentially growing volume with depth, which
Euclidean geometry cannot embed without distortion but which hyperbolic
space embeds isometrically (Sarkar 2011). Combining the two lets the
encoder represent taxonomic hierarchies as near-geodesic layouts while the
decoder still speaks the native compositional language of the data.

## 8. Implementation pointers

| Math | Code |
|---|---|
| Tangent-space Gaussian encoder | `HyperbolicPhILRVAE.encode` (inherited) |
| Sampling + projection on the ball | `HyperbolicPhILRVAE.reparam_to_ball` |
| `logmap0` → Linear decoder | `HyperbolicPhILRVAE.decode` (audit fix D2) |
| `geoopt` manifold | `PoincareBallExact(c=curvature)` |
| Likelihood NLLs | reused from `PhILRVAE` |
| CLI entry point | `biomevae.cli.vae_train_hyp_philrvae` |
| Snakemake registry | `hyp-philrvae` in `workflow/rules/common.smk` |

## 9. Removed in this generation

| Removed class | Replacement |
|---|---|
| `HyperbolicPhILRNBVAE` | `HyperbolicPhILRVAE` with `likelihood="dirichlet_multinomial"` or `"dirichlet_tree_multinomial"` |
| `HyperbolicPhILRZINBVAE` | `HyperbolicPhILRVAE` with `likelihood="dirichlet_tree_multinomial"` (zeros are absorbed by the within-clade Dirichlet-Multinomial) |
