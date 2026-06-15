# Euclidean VAE Theory and Mathematical Formulation

This document gives a rigorous formulation of the baseline Euclidean VAE implemented in `src/biomevae/models/vae.py`, and the objective functions used by the standard training pipeline.

---

## 1. Setup and notation

Let:

- \(x\in\mathbb{R}^p\): preprocessed input vector (optionally log-transformed and/or standardized by the training pipeline),
- \(z\in\mathbb{R}^d\): latent variable,
- \(g_\phi\): encoder network,
- \(f_\theta\): decoder network.

The encoder outputs a diagonal Gaussian posterior:
\[
(\mu_\phi(x),\log\sigma_\phi^2(x)) = g_\phi(x),
\]
\[
q_\phi(z\mid x)=\mathcal{N}(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))).
\]
The prior is isotropic Gaussian:
\[
p(z)=\mathcal N(0,I_d).
\]

---

## 2. Encoder-decoder parameterization

For hidden activations \(h^{(1)},\dots,h^{(L)}\):
\[
h^{(\ell)} = \sigma\!\big(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\big),
\quad h^{(0)}=x,
\]
with optional layer norm and dropout between layers.

The encoder head parameterizes
\[
\mu = W_\mu h^{(L)} + b_\mu,
\qquad
\log\sigma^2 = W_{\log\sigma^2} h^{(L)} + b_{\log\sigma^2}.
\]

Sampling is via reparameterization:
\[
z = \mu + \sigma\odot\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I_d),
\qquad \sigma = \exp\!\left(\tfrac12\log\sigma^2\right).
\]

Decoder outputs reconstruction:
\[
\hat x = f_\theta(z).
\]

---

## 3. KL divergence term

For diagonal Gaussian posterior vs standard normal prior:
\[
\mathrm{KL}\big(q_\phi(z\mid x)\|p(z)\big)
=\frac12\sum_{j=1}^d\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
\]

In code form with log-variance:
\[
\mathrm{KL}= -\tfrac12\sum_{j=1}^d\bigl(1+\log\sigma_j^2-\mu_j^2-\exp(\log\sigma_j^2)\bigr).
\]

These are algebraically identical.

---

## 4. Reconstruction losses

The training stack supports:

1. **MSE**
\[
\mathcal L_{\mathrm{rec}}^{\mathrm{MSE}}(x,\hat x)=\frac1p\sum_{i=1}^p(\hat x_i-x_i)^2.
\]

2. **MAE**
\[
\mathcal L_{\mathrm{rec}}^{\mathrm{MAE}}(x,\hat x)=\frac1p\sum_{i=1}^p|\hat x_i-x_i|.
\]

3. **Huber** (threshold \(\delta\))
\[
\mathcal L_{\mathrm{rec}}^{\mathrm{Huber}}(x,\hat x)
=\frac1p\sum_{i=1}^p
\begin{cases}
\tfrac12(\hat x_i-x_i)^2, & |\hat x_i-x_i|\le\delta,\\
\delta\left(|\hat x_i-x_i|-\tfrac12\delta\right), & \text{otherwise}.
\end{cases}
\]

---

## 5. Training objectives

### 5.1 \(\beta\)-VAE objective
\[
\mathcal J_\beta(x)
=\mathcal L_{\mathrm{rec}}(x,\hat x)
+\beta_t\,\mathrm{KL}\big(q_\phi(z\mid x)\|p(z)\big).
\]

Warmup schedule:
\[
\beta_t=\min\!\left(\beta_{\max},\beta_{\max}\frac{t}{T_{\mathrm{warmup}}}\right).
\]

### 5.2 Capacity objective
Used in the code path supporting controlled information bottleneck:
\[
\mathcal J_{\mathrm{cap}}(x)
=\mathcal L_{\mathrm{rec}}(x,\hat x)
+\gamma\,\left|\mathrm{KL}(q_\phi\|p)-C_t\right|,
\]
where \(C_t\) is a scheduled target capacity.

### 5.3 Free-bits stabilization
Per-dimension KL values may be lower-bounded to reduce posterior collapse:
\[
\mathrm{KL}_{\mathrm{fb}} = \sum_{j=1}^d\max\{\mathrm{KL}_j,\lambda_{\mathrm{fb}}\}.
\]

---

## 6. ELBO interpretation

The exact negative ELBO is
\[
-\mathcal{L}_{\mathrm{ELBO}}
= -\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+\mathrm{KL}(q_\phi\|p).
\]

When reconstruction is MSE/MAE/Huber, training can be interpreted as optimizing a surrogate likelihood term, with the KL part still exact for the Gaussian posterior/prior pair.

---

## 7. Diagram (computational graph)

```mermaid
flowchart LR
    X[Input x] --> ENC[Encoder MLP g_phi]
    ENC --> MU[mu_z]
    ENC --> LV[logvar_z]
    MU --> REP[Reparameterization]
    LV --> REP
    REP --> Z[z]
    Z --> DEC[Decoder MLP f_theta]
    DEC --> XH[x_hat]
    X --> REC[Reconstruction loss]
    XH --> REC
    MU --> KL[KL to N(0,I)]
    LV --> KL
    REC --> LOSS[Total objective]
    KL --> LOSS
```

---

## 8. Practical implications

- This model is the simplest baseline and easiest to optimize.
- It does **not** impose compositional simplex constraints by itself.
- It does **not** enforce tree/hierarchy consistency by construction.
- It is ideal as a control model to quantify gains from taxonomy-aware and count-aware variants.
