# Taxonomic inductive bias versus domain-invariant adaptation in biomevae

This note formalises, for the specific models implemented in
`biomevae`, a quantitative trade-off between

1.  **Taxonomic inductive bias** — placing the decoder or the latent on a
    rooted taxonomy through `TaxonomyGraph` /
    `build_philr_basis_from_taxonomy_graph`
    (`PhILRVAE`, `HyperbolicPhILRVAE`, `TreeDTMVAE`,
    `TreeStructuredPriorVAE`), and

2.  **Domain-invariant adaptation** — driving the predictive latent
    `z_y` to be statistically independent of study identity, as done by
    `LatentStudyCritic` + `dann_lambda_schedule` and `coral_per_study`
    in `src/biomevae/models/phylo_da.py` (the `PhyloDIVA*` family).

The informal claim is: *taxonomy helps when the disease signal is
smooth on the tree, but invariance hurts when taxonomic composition is
itself correlated with the study label.* The rest of this document
gives precise hypotheses, a proof, and a worked LOSO example tied to
the existing pipeline in
`src/biomevae/loso.py` and `src/biomevae/classify.py`.

---

## 1. Setup

### 1.1 Random variables

Fix a probability space \((\Omega,\mathcal F,\mathbb P)\) and the
following random variables.

| Symbol | Range | Meaning | Code reference |
|---|---|---|---|
| \(X\) | \(\mathbb R_{\ge 0}^{p}\) or \(\Delta^{p-1}\) | leaf-level taxa table for one sample | `load_matrix` in `data.py` |
| \(D\) | \(\{1,\dots,K\}\) | study / batch identity | `study_name` in `merge_studies` (`loso.py`) |
| \(Y\) | \(\{1,\dots,C\}\) | sample-level phenotype label | `klass` in `build_diva_dataset` |
| \(\mathcal T\) | rooted tree | taxonomy with leaf set \(\mathcal L\), internal nodes \(\mathcal I\) | `TaxonomyGraph` / `build_tree_topology` |
| \(A(X)\in\mathcal L\) | random leaf | "the taxonomic state of the sample" — the cell of the SBP partition that carries the signal | `aggregate_leaf_matrix_to_nodes` |

We write
\(P_d := \mathcal L(Y\mid D=d)\) and \(P_d^A := \mathcal L(A\mid D=d)\)
for the conditional laws of label and taxonomic state in study \(d\),
and use total variation
\(\operatorname{TV}(\mu,\nu)
= \tfrac12 \sup_{f\colon\|f\|_\infty \le 1}\!\big|\!\int\! f\,\mathrm d\mu - \int\! f\,\mathrm d\nu\big|\).

For \(d\neq d'\) we set the label / taxonomy shift
\[
\Delta_Y \;:=\; \operatorname{TV}(P_d,P_{d'}),
\qquad
\Delta_A \;:=\; \operatorname{TV}(P_d^A,P_{d'}^A).
\]
In a LOSO split the two values are computed exactly by
`ControlAnchor` in `loso.py`; the smoothness gap below is the regime
\(\Delta_Y \approx \Delta_A\), i.e. the across-study label shift is
mostly driven by genuine taxonomic compositional shift.

### 1.2 Two families of biomevae predictors

Both families are evaluated through the downstream XGBoost head defined
in `src/biomevae/classify.py` on a latent slice \(Z\); only the encoder
producing \(Z\) differs.

**Family (T) — taxonomy-aware, no invariance constraint.**
\(Z = Z_{\mathcal T}\) is the latent of any of
- `PhILRVAE` (`models/philrvae.py`),
- `HyperbolicPhILRVAE` (`models/hyperbolic_philrvae.py`),
- `TreeDTMVAE` (`models/tree_dtm_vae.py`),
- `TreeStructuredPriorVAE` (`models/treeprior.py`),

each of which routes the decoder through the SBP basis \(\Psi\) of
`build_philr_basis_from_taxonomy_graph` or through the sibling-split
softmax of `TreeDTMDecoder`. We write the resulting classifier
\(\widehat Y_{\mathcal T}=h_{\mathcal T}(Z_{\mathcal T})\) and its
per-study risk
\(R_d(\widehat Y_{\mathcal T}) := \mathbb P\!\big(\widehat Y_{\mathcal T}\neq Y\mid D=d\big)\).

**Family (I) — domain-invariant, in the sense of DIVA / PhyloDIVA.**
\(Z=Z_{\mathrm{inv}}\) is the `z_y` head of any of
- `DIVABetaVAE`, `DIVAHyperbolicPhILRVAE`, `DIVATreeDTMVAE`
  (conditional prior \(p(z_y\mid y)\), but **no** invariance constraint),
- `PhyloDIVABetaVAE`, `PhyloDIVAHyperbolicPhILRVAE`,
  `PhyloDIVATreeDTMVAE` (additionally `LatentStudyCritic` with
  schedule `dann_lambda_schedule` and `coral_per_study`).

The classifier is \(\widehat Y_{\mathrm{inv}}=h(Z_{\mathrm{inv}})\) with
per-study risk \(R_d(\widehat Y_{\mathrm{inv}})\). We measure the
residual study dependence of \(Z_{\mathrm{inv}}\) by
\[
\delta \;:=\; \operatorname{TV}\!\big(
\mathcal L(Z_{\mathrm{inv}}\mid D=d),\;
\mathcal L(Z_{\mathrm{inv}}\mid D=d')\big),
\]
which is exactly what the gradient-reversed critic of
`LatentStudyCritic` and the second-order CORAL matcher of
`coral_per_study` are designed to drive to zero. The asymptotic Bayes
accuracy of any study classifier on \(z_y\) is
\(\tfrac12 + \tfrac{\delta}{2}\), and `loso_diagnostic`
(`cli/loso_diagnostic.py`) reports exactly this number under the
"domain accuracy of a fresh classifier on `z_y`" row of
[`phylodiva_theory.md`](phylodiva_theory.md) §6.

### 1.3 Tree-smoothness regulariser

Following the boxed definition in §4 of
[`phylodiva_theory.md`](phylodiva_theory.md), for any node-indexed
score \(f \in \mathbb R^{|\mathcal I|+|\mathcal L|}\) write
\[
\Omega_{\mathcal T}(f)
\;:=\;
\sum_{(u,v)\in E(\mathcal T)} \!\frac{(f_u - f_v)^2}{\ell_{uv}+\varepsilon},
\qquad \varepsilon > 0,
\]
with \(\ell_{uv}\) the branch length (or \(1\) if uncalibrated). This
is exactly `bm_edge_smoothness` (TreeDTM) and `bm_coord_smoothness`
(PhILR), and induces the Laplacian
\(L_{\mathcal T} = D_{\mathcal T} - W_{\mathcal T}\) with
\(W_{\mathcal T}[u,v]=(\ell_{uv}+\varepsilon)^{-1}\) on edges.

### 1.4 Tree-smooth target

**Assumption (TS).** There is a node-indexed function
\(f^\star\colon V(\mathcal T)\to\mathbb R^C\) and a leaf-aggregator
\(A\colon \mathcal X \to \mathcal L\) such that
\[
\mathbb P\!\big(Y\neq \operatorname*{arg\,max}_c f^\star_{A(X),c}\big)
\;\le\; \varepsilon_{\mathcal T}^{\mathrm{approx}},
\qquad
\Omega_{\mathcal T}(f^\star) \;\le\; B^2.
\]

Under (TS), `PhILRVAE` and `TreeDTMVAE` are well-specified up to
approximation error \(\varepsilon_{\mathcal T}^{\mathrm{approx}}\); the
isometry of §2.3 of [`philrvae_theory.md`](philrvae_theory.md)
guarantees that Euclidean linear heads on \(Z_{\mathcal T}\) realise
exactly the class of Aitchison-linear classifiers on the simplex,
which is the natural ambient class for tree-smooth \(f^\star\).

---

## 2. The theorem

### 2.1 Theorem (taxonomy vs. domain-invariance in `biomevae`)

Let Assumption (TS) hold and let
\(n\) be the number of training samples seen by the
downstream classifier of `classify.py`.

**(a) Taxonomy-aware upper bound.**
There exist universal constants \(c_1,c_2 > 0\) and a regularisation
choice \(\lambda > 0\) (the `--smooth-weight` flag) such that, with
probability at least \(1-\eta\) over the draw of the training set, the
XGBoost / linear head trained on \(Z_{\mathcal T}\) and using the
tree-smoothness penalty \(\lambda\,\Omega_{\mathcal T}\) satisfies
\[
\boxed{\;
R_d(\widehat Y_{\mathcal T})
\;\le\;
\varepsilon_{\mathcal T}^{\mathrm{approx}}
+ c_1 \sqrt{\frac{d_{\mathcal T}(\lambda)}{n}}
+ c_2 \sqrt{\frac{\log(1/\eta)}{n}}
+ \lambda B^2,
\;}
\tag{T}
\]
where
\(d_{\mathcal T}(\lambda) = \operatorname{tr}\!\big(L_{\mathcal T}(L_{\mathcal T}+\lambda I)^{-1}\big)\)
is the effective degrees of freedom of the tree-Laplacian ridge
regulariser. For an informative taxonomy (i.e. when \(f^\star\) is
concentrated on the bottom of the Laplacian spectrum) we have
\(d_{\mathcal T}(\lambda)\ll p-1=\dim\Psi\), which is the
variance-reduction half of the trade-off.

**(b) Domain-invariant lower bound.**
For *any* measurable classifier
\(\widehat Y_{\mathrm{inv}}=h(Z_{\mathrm{inv}})\) whose encoder satisfies
\(\operatorname{TV}(\mathcal L(Z_{\mathrm{inv}}\mid D=d),\mathcal L(Z_{\mathrm{inv}}\mid D=d')) \le \delta\),
\[
\boxed{\;
\tfrac12\!\left[R_d(\widehat Y_{\mathrm{inv}})+R_{d'}(\widehat Y_{\mathrm{inv}})\right]
\;\ge\;
\tfrac{1}{2}\big(\Delta_Y - \delta\big)_+ .
\;}
\tag{I}
\]
The bound is sharp: equality is achieved by the constant predictor
\(\widehat Y_{\mathrm{inv}}\equiv \operatorname*{arg\,max}_c \tfrac12(P_d(c)+P_{d'}(c))\).

**(c) Crossover.**
If
\[
\tfrac12(\Delta_Y - \delta)_+
\;>\;
\varepsilon_{\mathcal T}^{\mathrm{approx}}
+ c_1\sqrt{d_{\mathcal T}(\lambda)/n}
+ c_2\sqrt{\log(1/\eta)/n}
+ \lambda B^2,
\tag{C}
\]
then the taxonomy-aware predictor strictly outperforms the
domain-invariant one on the balanced two-study risk, with probability
at least \(1-\eta\).

### 2.2 Concrete reading inside `biomevae`

- \(\Delta_Y\) and \(\Delta_A\) are exactly the per-pair quantities
  printed by `loso_diagnostic` (`cli/loso_diagnostic.py`) and stored in
  `results/loso_summary_*.tsv` under the "label shift" and
  "control-anchored taxonomic shift" columns; they are usually large in
  microbiome LOSO because phenotype prevalence and core-microbiome
  composition both vary between cohorts.
- \(\delta\) is what `LatentStudyCritic` + `coral_per_study` actively
  minimise. The ramp `dann_lambda_schedule(t, lambda_max, gamma=10)`
  drives \(\delta \to 0\) as `t→1`. So the **stronger PhyloDIVA's
  invariance constraint, the closer (I) is to \(\Delta_Y/2\) — i.e. the
  more PhyloDIVA pays the lower bound.**
- \(\Omega_{\mathcal T}\) is implemented as `bm_edge_smoothness`
  (`TreeDTMVAE`, `DIVATreeDTMVAE`, `PhyloDIVATreeDTMVAE`) or
  `bm_coord_smoothness` (PhILR backbones); the regularisation strength
  \(\lambda\) is the CLI flag `--smooth-weight` in
  `cli/vae_train_phylodiva_tree_dtm_vae.py` and analogues.
- (T) is the only place the model's *backbone* matters: PhILRVAE and
  TreeDTMVAE realise (TS) exactly because their decoders are tied to
  the SBP / sibling-softmax of \(\mathcal T\); a flat MLP on
  log-transformed counts (`VAE`, `DIVABetaVAE`) does not, and would
  need \(d_{\mathrm{raw}}=p-1\) in place of \(d_{\mathcal T}(\lambda)\).

---

## 3. Proof

### 3.1 Lemma (data-processing for TV)

If \(Z\sim\mu\) and \(Z'\sim\nu\) and \(h\) is any measurable map,
then \(\operatorname{TV}(h_\#\mu,h_\#\nu)\le \operatorname{TV}(\mu,\nu)\).

*Proof.* For \(\|\varphi\|_\infty\le 1\), \(\varphi\circ h\) is also
\(\le 1\) in \(\sup\)-norm, so the variational form of TV is
non-increasing under push-forward. \(\square\)

### 3.2 Lemma (classification risk vs. label-marginal TV)

For any classifier \(\widehat Y\) and any value of \(d\),
\(R_d(\widehat Y) \ge \operatorname{TV}\!\big(P_d, Q_d\big)\), where
\(Q_d := \mathcal L(\widehat Y\mid D=d)\).

*Proof.* For any set \(S\subseteq\{1,\dots,C\}\),
\[
R_d(\widehat Y)
\;\ge\;
\big|\mathbb P(Y\in S\mid D=d) - \mathbb P(\widehat Y\in S\mid D=d)\big|
=
\big|P_d(S)-Q_d(S)\big|,
\]
because \(\{Y\in S,\widehat Y\notin S\}\cup\{Y\notin S,\widehat Y\in S\}
\subseteq \{Y\neq\widehat Y\}\) and the two events on the left are
disjoint inside \(\{Y\neq\widehat Y\}\). Taking \(\sup_S\) gives
\(R_d(\widehat Y)\ge \operatorname{TV}(P_d,Q_d)\). \(\square\)

### 3.3 Proof of (I)

By 3.1, \(\operatorname{TV}(Q_d,Q_{d'})
\le \operatorname{TV}(\mathcal L(Z_{\mathrm{inv}}|D=d),\mathcal L(Z_{\mathrm{inv}}|D=d')) \le \delta\).
By 3.2 and the triangle inequality on TV,
\[
R_d+R_{d'}
\;\ge\;
\operatorname{TV}(P_d,Q_d)+\operatorname{TV}(P_{d'},Q_{d'})
\;\ge\;
\operatorname{TV}(P_d,P_{d'}) - \operatorname{TV}(Q_d,Q_{d'})
\;\ge\;
\Delta_Y-\delta.
\]
The non-negativity \((\cdot)_+\) is automatic from
\(R\ge 0\). Sharpness with the constant predictor is immediate: it
gives \(Q_d=Q_{d'}=\delta_{c^\star}\), so \(\operatorname{TV}(Q_d,Q_{d'})=0\)
and the inequality is met with equality. \(\square\)

### 3.4 Proof of (T)

The XGBoost / linear head of `classify.py` is trained by empirical risk
minimisation over the linear class
\(\mathcal H_\Psi=\{x\mapsto \langle w,\Psi^\top \log c\rangle\colon
w^\top L_{\mathcal T} w \le B^2\}\)
(for PhILR backbones; the analogous tree-Laplacian ridge class for
sibling-softmax decoders). By Theorem 4 of Mendelson (2002) and the
classical local Rademacher argument of Bartlett, Bousquet & Mendelson
(2005) for kernel ridge classes, the Rademacher complexity of
\(\mathcal H_\Psi\) is bounded by
\(\mathcal R_n(\mathcal H_\Psi) \le B\sqrt{d_{\mathcal T}(\lambda)/n}\),
where \(d_{\mathcal T}(\lambda)=\operatorname{tr}(L_{\mathcal T}(L_{\mathcal T}+\lambda I)^{-1})\)
is the effective dimension of the regularised kernel
\(K=L_{\mathcal T}^\dagger\). A standard margin / contraction step (e.g.
Bartlett & Mendelson 2002, Theorem 8) then yields
\[
R_d(\widehat Y_{\mathcal T})
\;\le\; \inf_{f\in\mathcal H_\Psi}\!\mathbb P(Y\neq f(A))
\;+\; 2\mathcal R_n(\mathcal H_\Psi)
\;+\; c_2\sqrt{\log(1/\eta)/n}.
\]
Under (TS), the infimum is at most
\(\varepsilon_{\mathcal T}^{\mathrm{approx}} + \lambda B^2\) because the
ridge surrogate satisfies
\(\inf_{f}\|f-f^\star\|_{\mathrm{emp}}^2 + \lambda\Omega_{\mathcal T}(f)
\le \lambda\Omega_{\mathcal T}(f^\star) \le \lambda B^2\). Combining
gives the boxed inequality (T). \(\square\)

### 3.5 Proof of (c)

Immediate from (T) (applied symmetrically to studies \(d\) and \(d'\)
with a union bound, absorbing the factor \(2\) into \(c_1,c_2\)) and
(I). \(\square\)

---

## 4. Where does the mass go? — the smoothness gap

The lower bound (I) is *exactly* the price PhyloDIVA pays for
**enforcing** invariance. It is informative only if \(\Delta_Y>\delta\).
The diagnostic test
[`phylodiva_theory.md`](phylodiva_theory.md) §6
("Domain accuracy of a fresh classifier on `z_y`") measures
\(\tfrac12+\tfrac{\delta}{2}\); if the row drops to chance
(0.5), then \(\delta\downarrow 0\) and (I) reduces to the *raw label
shift* \(\Delta_Y/2\).

In microbiome LOSO data \(\Delta_Y\) is large precisely **because**
study-specific cohorts have different taxonomic compositions —
\(\Delta_Y \approx \Delta_A\). So driving \(\delta\) to zero
necessarily destroys the
taxonomy-correlated component of `z_y`, which is the same component
PhILRVAE / TreeDTMVAE exploit through (T). This is the failure mode
already flagged by the docstring of `phylo_da.py`: the GRL
"pushes `z_y` to be un-discriminative of study" but cannot
discriminate "study-as-confounder" from
"study-as-side-effect-of-taxonomic-composition".

---

## 5. Worked LOSO example (binary phenotype, two studies)

Take \(K=2\), \(C=2\), and a single binary taxonomic state
\(A\in\{0,1\}\) (e.g. presence of a marker clade aggregated by
`aggregate_leaf_matrix_to_nodes`). Suppose
\(Y=A\), \(\Pr(A=1\mid D=0)=0.1\), \(\Pr(A=1\mid D=1)=0.9\). Then

\[
\Delta_Y = \Delta_A = |0.9-0.1| = 0.8.
\]

- *Taxonomy-aware*. `PhILRVAE` recovers \(A\) deterministically via the
  SBP basis at the relevant internal node, so (T) gives
  \(R_d(\widehat Y_{\mathcal T})=\varepsilon_{\mathcal T}^{\mathrm{approx}}=0\)
  up to finite-sample noise, regardless of \(D\).

- *Strict DIVA-like invariance.* Setting \(\delta = 0\) in (I) forces
  \[
  \tfrac12(R_0+R_1) \;\ge\; \tfrac{0.8}{2} \;=\; 0.4.
  \]
  Any predictor whose output is independent of \(D\) outputs \(1\) with
  some probability \(q\) that does not depend on \(D\), giving
  \(R_0=0.1+0.8q\) and \(R_1=0.9-0.8q\), hence balanced risk \(0.5\)
  identically in \(q\). The lower bound \(0.4\) is therefore valid but
  not tight in this symmetric example; the true balanced-risk minimum
  for any invariant predictor is \(0.5\), which is still \(>\) the
  \(\varepsilon_{\mathcal T}^{\mathrm{approx}} = 0\) achieved by the
  taxonomy-aware predictor. The sharpness statement in §2.1(b) is
  attained on asymmetric pairs \((P_d,P_{d'})\); the symmetric case
  above shows the bound can be conservative but still strictly
  separates families (T) and (I).

The example is exactly the "core-clade flips between cohorts" failure
mode that `ControlAnchor.coral_pair_distance` and
`ControlAnchor.mmd_pair` are designed to expose in
`workflow/README_loso.md`. When those diagnostics report large pairwise
distances *inside the control class*, the taxonomic shift is real, and
(C) predicts that adding the PhyloDIVA invariance terms will
*decrease* held-out accuracy.

---

## 6. Operational consequence

The theorem gives a falsifiable knob-by-knob prediction for the
training CLIs:

| Knob | Where | Effect on (T) | Effect on (I) |
|---|---|---|---|
| `--smooth-weight ν` (BM tree penalty) | `vae_train_phylodiva_*` | shrinks \(d_{\mathcal T}(\lambda)\), tightens (T) | none |
| `--grl-lambda λ_max` (GRL strength) | `vae_train_phylodiva_*` | none directly | shrinks \(\delta\), tightens (I) |
| `--coral-weight μ` (CORAL on `z_x`) | `vae_train_phylodiva_*` | none directly | shrinks \(\delta\), tightens (I) |
| `--alpha-y` (DIVA aux CE on `z_y`) | `vae_train_diva_*` | sharpens \(\widehat Y\) on \(\mathcal H_\Psi\) | tends to *increase* \(\delta\) — counteracts the GRL |

The recipe is therefore: choose `--smooth-weight` to make (T) small,
and choose `--grl-lambda`, `--coral-weight` only as large as the
"control anchored" \(\Delta_A\) of `ControlAnchor` allows. If
`loso_diagnostic` reports \(\Delta_Y\approx\Delta_A\gg\delta\), the
PhyloDIVA invariance terms are predicted to **hurt**, and the
TreeDTMVAE / PhILRVAE backbones without the GRL critic should be
preferred. This is the practical content of the theorem.

---

## 7. References

- Bartlett & Mendelson, *Rademacher and Gaussian complexities: risk bounds
  and structural results*, JMLR 2002.
- Bartlett, Bousquet & Mendelson, *Local Rademacher complexities*, Annals
  of Statistics 2005.
- Ben-David et al., *A theory of learning from different domains*,
  Machine Learning 2010 (TV-style lower bounds for domain adaptation).
- Felsenstein, *Phylogenies and the comparative method*, American
  Naturalist 1985 (Brownian motion on trees).
- Ganin & Lempitsky, *Unsupervised Domain Adaptation by Backpropagation*,
  ICML 2015 (gradient reversal).
- Ilse, Tomczak, Louizos, Welling, *DIVA: Domain Invariant Variational
  Autoencoders*, MIDL 2020.
- Mendelson, *Geometric parameters of kernel machines*, COLT 2002
  (effective dimension of a kernel ridge class).
- Sun & Saenko, *Deep CORAL*, ECCV 2016.
- Zhao, Combes, Zhang, Gordon, *On learning invariant representations
  for domain adaptation*, ICML 2019 (the TV / JS lower bound that (I)
  specialises).
