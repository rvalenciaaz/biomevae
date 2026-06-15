# DIVA — Domain-Invariant VAE (backbone-agnostic)

This document describes the **DIVA** building blocks implemented in
`src/biomevae/models/diva.py` and the three concrete biomevae models that
wrap them:

| Wrapper | Backbone | File |
|---|---|---|
| `DIVABetaVAE` | β-VAE on raw / log-transformed counts | `models/diva_betavae.py` |
| `DIVAHyperbolicPhILRVAE` | Hyperbolic PhILR (compositional) | `models/diva_hyp_philrvae.py` |
| `DIVATreeDTMVAE` | Tree-Dirichlet-Tree-Multinomial | `models/diva_treedtmvae.py` |

DIVA itself is **likelihood-agnostic**: the reconstruction terms come from
the backbones — see [`vae_theory.md`](vae_theory.md),
[`hyperbolic_philrvae_theory.md`](hyperbolic_philrvae_theory.md), and
[`tree_dtm_vae_theory.md`](tree_dtm_vae_theory.md) — and DIVA contributes
only the latent partitioning, conditional priors, KL terms and auxiliary
classifier losses.

Reference: Ilse, Tomczak, Louizos, Welling, *DIVA: Domain Invariant Variational
Autoencoders*, MIDL 2020.

---

## 1. Latent partition

DIVA factorises the latent into three independent Gaussian blocks:

\[
\mathbf z = [\,\mathbf z_d \,;\, \mathbf z_y \,;\, \mathbf z_x\,],
\qquad
\mathbf z_d \in \mathbb R^{d_d},\; \mathbf z_y \in \mathbb R^{d_y},\; \mathbf z_x \in \mathbb R^{d_x},
\]

with **conditional** priors

\[
\begin{aligned}
p(\mathbf z_d \mid d) &= \mathcal N\!\big(\boldsymbol\mu_d(d),\; \mathrm{diag}\,\boldsymbol\sigma_d^2(d)\big), \\
p(\mathbf z_y \mid y) &= \mathcal N\!\big(\boldsymbol\mu_y(y),\; \mathrm{diag}\,\boldsymbol\sigma_y^2(y)\big), \\
p(\mathbf z_x)        &= \mathcal N(\mathbf 0, I_{d_x}),
\end{aligned}
\]

where \(d \in \{1,\dots,D\}\) indexes the **domain** (typically the study)
and \(y \in \{1,\dots,Y\}\) indexes the **class label** (e.g. disease
status). Each conditional prior is parameterised as a lookup
table — `CategoryConditionalPrior` — implemented as a pair of
`nn.Embedding` tables so the prior is exactly per-category, with no MLP and
no inter-category interpolation.

## 2. Encoder

A shared trunk \(h = f_\phi(\mathbf x)\) produces a feature vector. Three
**independent** Gaussian heads (`DIVAEncoderHeads`) project it to the three
latent factors:

\[
(\boldsymbol\mu_d, \log\boldsymbol\sigma_d^2) = \mathrm{Linear}_d(h),
\quad
(\boldsymbol\mu_y, \log\boldsymbol\sigma_y^2) = \mathrm{Linear}_y(h),
\quad
(\boldsymbol\mu_x, \log\boldsymbol\sigma_x^2) = \mathrm{Linear}_x(h),
\]

with a constant negative `logvar_bias` (default `-2.0`) so the encoder is
biased toward small variance at the start of training and the conditional
priors are not "absorbed" by the warm-up KL.

Reparameterised sampling is the standard tangent-space Gaussian draw

\[
\mathbf z_\bullet = \boldsymbol\mu_\bullet + \boldsymbol\sigma_\bullet \odot \boldsymbol\varepsilon,
\quad
\boldsymbol\varepsilon \sim \mathcal N(\mathbf 0, I),
\]

separately for each of \(\bullet \in \{d, y, x\}\). Hyperbolic backbones
push `z_d`, `z_y` and `z_x` to the ball via `expmap_0` (see
[`hyperbolic_philrvae_theory.md`](hyperbolic_philrvae_theory.md)).

## 3. Auxiliary classifiers

Two small MLP heads (`AuxClassifier`) classify the side information from the
matching latent factor:

\[
q(d \mid \mathbf z_d) = \mathrm{softmax}\,\psi_d(\mathbf z_d),
\qquad
q(y \mid \mathbf z_y) = \mathrm{softmax}\,\psi_y(\mathbf z_y).
\]

These push \(\mathbf z_d\) to *encode* the domain and \(\mathbf z_y\) to
*encode* the class, complementing the conditional priors which pull each
factor *toward* the category-specific mean.

## 4. ELBO

For a fully labelled sample \((\mathbf x, d, y)\):

\[
\mathcal L_\text{DIVA}^{(i)}
=
\underbrace{-\log p_\theta(\mathbf x^{(i)} \mid \mathbf z^{(i)})}_{\text{backbone NLL}}
+
\beta_t \Big(
\mathrm{KL}_d^{(i)} + \mathrm{KL}_y^{(i)} + \mathrm{KL}_x^{(i)}
\Big)
+ \alpha_d \,\mathrm{CE}_d^{(i)}
+ \alpha_y \,\mathrm{CE}_y^{(i)},
\]

with

\[
\begin{aligned}
\mathrm{KL}_d &= \mathrm{KL}\!\big(q_\phi(\mathbf z_d \mid \mathbf x) \,\|\, p(\mathbf z_d \mid d)\big),\\
\mathrm{KL}_y &= \mathrm{KL}\!\big(q_\phi(\mathbf z_y \mid \mathbf x) \,\|\, p(\mathbf z_y \mid y)\big),\\
\mathrm{KL}_x &= \mathrm{KL}\!\big(q_\phi(\mathbf z_x \mid \mathbf x) \,\|\, \mathcal N(\mathbf 0, I_{d_x})\big),\\
\mathrm{CE}_d &= -\log q(d \mid \mathbf z_d),\\
\mathrm{CE}_y &= -\log q(y \mid \mathbf z_y).
\end{aligned}
\]

Closed-form Gaussian KL between two diagonal Gaussians:

\[
\mathrm{KL}\!\big(\mathcal N(\boldsymbol\mu_q, \boldsymbol\sigma_q^2)\,\|\,\mathcal N(\boldsymbol\mu_p, \boldsymbol\sigma_p^2)\big)
= \tfrac12 \sum_j \!\Bigg(
\frac{\sigma_{q,j}^2 + (\mu_{q,j} - \mu_{p,j})^2}{\sigma_{p,j}^2}
- 1 + \log \frac{\sigma_{p,j}^2}{\sigma_{q,j}^2}
\Bigg).
\]

Implemented in `gaussian_kl` (`diva.py`); a `free_bits` floor per dimension
can be applied to all three KL terms.

\(\beta_t\) follows the same linear warm-up as the other biomevae models
(`losses.beta_schedule`); the auxiliary CE weights \(\alpha_d, \alpha_y\)
are constants exposed as CLI flags (`--alpha-d`, `--alpha-y`).

### 4.1 Semi-supervised mode (missing class labels)

When a sample has no class label (`klass = -1`), the `kl_y` term cannot use
the class-conditional prior. The implementation falls back to the
**marginal-of-prior** — the per-dimension mean of the embedding tables —
weighted by the supervised / unsupervised split sizes:

\[
\mathrm{KL}_y^\text{batch}
= \frac{1}{B}\!\left(
\sum_{i:\,y^{(i)} \neq -1}\! \mathrm{KL}\!\big(q^{(i)} \| p(\mathbf z_y \mid y^{(i)})\big)
+
\sum_{i:\,y^{(i)} = -1}\! \mathrm{KL}\!\big(q^{(i)} \| \bar p_y\big)
\right),
\]

with
\(\bar p_y = \mathcal N\!\big(\overline{\boldsymbol\mu_y}, \overline{\mathrm{diag}\,\boldsymbol\sigma_y^2}\big)\)
the mean of the conditional priors over classes. The auxiliary
cross-entropy `CE_y` is masked to the labelled subset and the caller
guards with `n_y_labelled > 0` before adding it to the loss.

In our LOSO setting every sample has a class label, so this branch is
quiescent; it is kept for full DIVA correctness on partially labelled data.

## 5. Why the partition works

- The conditional prior \(p(\mathbf z_d \mid d)\) *pulls* the matching
  domain code toward the domain-specific mean — different studies live in
  different regions of \(\mathbf z_d\)-space.
- The auxiliary classifier \(q(d \mid \mathbf z_d)\) *pushes* \(\mathbf z_d\)
  to be linearly informative about the domain.
- The reconstruction term has access to all three blocks, so anything that
  is unrelated to either domain or class is forced into \(\mathbf z_x\),
  which has the standard \(\mathcal N(\mathbf 0, I)\) prior and the highest
  capacity. Downstream classifiers should read \(\mathbf z_y\) only.

## 6. Per-backbone wrappers

Each wrapper inherits the corresponding backbone's reconstruction
likelihood and adds a `DIVALoss` head:

| Wrapper | Reconstruction likelihood | Notes |
|---|---|---|
| `DIVABetaVAE` | Gaussian on log-transformed counts (β-VAE) | Smallest model; useful for ablation. |
| `DIVAHyperbolicPhILRVAE` | Compositional (`philr_gaussian` / Dirichlet-Multinomial / Dirichlet-Tree-Multinomial) on Poincaré ball | Each of `z_d, z_y, z_x` is sampled separately in tangent space and `expmap_0`-projected. |
| `DIVATreeDTMVAE` | Tree-Multinomial / Dirichlet-Tree-Multinomial / Dirichlet-Tree | Likelihood reads node-aggregated tensor; encoder uses sibling-centred log-ratios. |

## 7. Implementation correspondence

| Math | Code |
|---|---|
| Encoder heads \(\boldsymbol\mu_\bullet, \log\boldsymbol\sigma_\bullet^2\) | `DIVAEncoderHeads.forward` |
| Conditional priors \(p(z_\bullet \mid c)\) | `CategoryConditionalPrior` |
| Auxiliary classifiers \(q(d|z_d), q(y|z_y)\) | `AuxClassifier` |
| Gaussian-Gaussian KL | `gaussian_kl` |
| Gaussian-standard-normal KL | `gaussian_kl_to_standard_normal` |
| Loss orchestrator | `DIVALoss.forward → DIVALossOutputs` |
| CLI entry points | `biomevae-train-diva-betavae`, `biomevae-train-diva-hyp-philrvae`, `biomevae-train-diva-tree-dtm` |

## 8. Diagnostic expectations

A correctly trained DIVA model should give:

- High accuracy of `q(d|z_d)` on held-out splits — confirms that `z_d`
  *captures* the domain.
- Near-chance accuracy of any classifier reading `z_y` for the **domain**
  label (DIVA does not enforce this, but PhyloDIVA does — see
  [`phylodiva_theory.md`](phylodiva_theory.md)).
- Near-chance accuracy of any classifier reading `z_x` for either domain or
  class.

Failure to scrub study fingerprints from `z_y` is the gap that motivates
PhyloDIVA's gradient-reversed study critic.
