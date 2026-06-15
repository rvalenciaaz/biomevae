# DeepPhyloFusionVAE Theory and Mathematical Formulation

This document formalizes `DeepPhyloFusionVAE` in `src/biomevae/models/phylo_fusion.py`.

## 1. Phylogenetic side information
Given fixed phylogenetic embedding matrix
\[
E_{\text{phy}}\in\mathbb R^{p\times d_p}
\]
(stored as `phylo_embeddings`), the model computes per-sample weighted summary from feature weights \(w\):
\[
\bar e(x)=\sum_{i=1}^p \tilde w_i E_{\text{phy},i},
\qquad
\tilde w_i=\frac{\max(w_i,0)}{\sum_j\max(w_j,0)}.
\]
(If denominator is zero, implementation uses a safe fallback denominator.)

## 2. Fusion encoder
Input to encoder MLP is concatenation
\[
h_0=[x;\bar e(x)]\in\mathbb R^{p+d_p}.
\]
Then
\[
(\mu,\log\sigma^2)=g_\phi(h_0),
\qquad q_\phi(z\mid x)=\mathcal N(\mu,\operatorname{diag}(\sigma^2)).
\]

## 3. Decoder and objective
Decoder is standard MLP:
\[
\hat x=f_\theta(z),\quad z=\mu+\sigma\odot\varepsilon,\; \varepsilon\sim\mathcal N(0,I).
\]
Training objective follows shared VAE loss:
\[
\mathcal J(x)=\mathcal L_{\text{rec}}(x,\hat x)+\beta_t\,\mathrm{KL}(q_\phi(z\mid x)\|\mathcal N(0,I)).
\]

## 4. Interpretation
This architecture injects phylogenetic context additively at encoder input level. It is a **late-fusion representation model**: learn latent structure from observed abundances and an externally precomputed phylogenetic embedding summary.

## 5. Diagram
```mermaid
flowchart LR
  X[x] --> W[weights for phylo summary]
  P[phylo_embeddings] --> W
  W --> S[summary e_bar]
  X --> CAT[concat x || e_bar]
  S --> CAT
  CAT --> ENC[encoder MLP]
  ENC --> MU[mu]
  ENC --> LV[logvar]
  MU --> Z[z]
  LV --> Z
  Z --> DEC[decoder MLP]
  DEC --> XH[x_hat]
```
