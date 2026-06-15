# TreeStructuredPriorVAE Theory and Mathematical Formulation

This document formalizes `TreeStructuredPriorVAE` in `src/biomevae/models/treeprior.py`.

## 1. Base encoder/decoder
The model inherits the graph-pooled encoder idea from `TaxonomyGraphVAE`:
\[
(\mu_q,\log\sigma_q^2)=g_\phi(x),\quad q_\phi(z\mid x)=\mathcal N(\mu_q,\operatorname{diag}(\sigma_q^2)),\quad \hat x=f_\theta(z).
\]

## 2. Learnable tree-conditioned Gaussian prior
Unlike a fixed standard prior, this model learns node-level prior parameters:
\[
\{\mu_v^{(p)},\log\sigma_{v}^{2,(p)}\}_{v=1}^{n_{\text{nodes}}}.
\]
For feature-selector indices \(S\), gather feature priors and mix by normalized sample weights \(w\):
\[
\mu_p(x)=\sum_{i=1}^p w_i\mu_{S_i}^{(p)},
\qquad
\sigma_p^2(x)=\sum_{i=1}^p w_i\sigma_{S_i}^{2,(p)}.
\]
Thus
\[
p(z\mid x)=\mathcal N\big(\mu_p(x),\operatorname{diag}(\sigma_p^2(x))\big).
\]

## 3. Branch smoothness regularization on prior parameters
For taxonomy edges \((u,v)\) with weight \(a_{uv}\), branch penalty is
\[
\mathcal R_{\text{branch}}
=\lambda_b\,\frac{1}{|E|}\sum_{(u,v)\in E}a_{uv}
\left(\|\mu_u^{(p)}-\mu_v^{(p)}\|_2^2+\|\log\sigma_u^{2,(p)}-\log\sigma_v^{2,(p)}\|_2^2\right).
\]

## 4. KL term with conditional diagonal-Gaussian prior
For diagonal Gaussian posterior \(q=\mathcal N(\mu_q,\sigma_q^2I)\) and conditional prior \(p=\mathcal N(\mu_p,\sigma_p^2I)\):
\[
\mathrm{KL}(q\|p)
=\frac12\sum_j\left[
\log\frac{\sigma_{p,j}^2}{\sigma_{q,j}^2}
+\frac{\sigma_{q,j}^2+(\mu_{q,j}-\mu_{p,j})^2}{\sigma_{p,j}^2}-1
\right].
\]

## 5. Full training objective
\[
\mathcal J(x)
=\mathcal L_{\text{rec}}(x,\hat x)
+\beta_t\,\mathrm{KL}\big(q_\phi(z\mid x)\|p(z\mid x)\big)
+\mathcal R_{\text{branch}}.
\]

## 6. Diagram
```mermaid
flowchart LR
  X[x] --> GENC[Graph-pooled encoder]
  GENC --> QMU[q mu]
  GENC --> QLV[q logvar]
  X --> W[normalized feature weights]
  W --> PMIX[mix node priors]
  PN[learned node prior params] --> PMIX
  PMIX --> PMU[p mu]
  PMIX --> PLV[p logvar]
  QMU --> KL[KL(q||p)]
  QLV --> KL
  PMU --> KL
  PLV --> KL
  QMU --> Z[z sample]
  QLV --> Z
  Z --> DEC[decoder]
  DEC --> XH[x_hat]
  X --> REC[reconstruction loss]
  XH --> REC
  PN --> BR[branch regularizer]
  REC --> LOSS[Total loss]
  KL --> LOSS
  BR --> LOSS
```
