# Theory Documentation Coverage (Model-by-Model)

This file verifies that each VAE model class under `src/biomevae/models/` has
corresponding mathematical / theory documentation in `docs/`.

The coverage tracks the **current** implementation tree. The previous
generation of count-likelihood models (`TreeNBVAE`, `HyperbolicPhILRZINBVAE`)
has been retired in favour of statistically appropriate compositional
likelihoods on the tree; the new DIVA and PhyloDIVA families add
domain-conditional latents and adversarial domain-adaptation on top of the
existing backbones.

## Coverage table — base backbones

| Model class | Implementation file | Markdown theory doc | LaTeX theory doc |
|---|---|---|---|
| `VAE` | `src/biomevae/models/vae.py` | `docs/vae_theory.md` | `docs/vae_theory.tex` |
| `HyperbolicVAE` | `src/biomevae/models/hyperbolic.py` | `docs/hyperbolicvae_theory.md` | `docs/hyperbolicvae_theory.tex` |
| `TaxonomyGraphVAE` | `src/biomevae/models/graph.py` | `docs/graphvae_theory.md` | `docs/graphvae_theory.tex` |
| `TreeStructuredPriorVAE` | `src/biomevae/models/treeprior.py` | `docs/treepriorvae_theory.md` | `docs/treepriorvae_theory.tex` |
| `DeepPhyloFusionVAE` | `src/biomevae/models/phylo_fusion.py` | `docs/phylofusionvae_theory.md` | `docs/phylofusionvae_theory.tex` |
| `FlowXFormerVAE` | `src/biomevae/models/flowxformer.py` | `docs/flowxformervae_theory.md` | `docs/flowxformervae_theory.tex` |
| `HGVAE_ZI` | `src/biomevae/models/hgvae_zi.py` | `docs/hgvae_zi_theory.md` | `docs/hgvae_zi_theory.tex` |
| `TreeDTMVAE` | `src/biomevae/models/tree_dtm_vae.py` | `docs/tree_dtm_vae_theory.md` | — |
| `PhILRVAE` | `src/biomevae/models/philrvae.py` | `docs/philrvae_theory.md` | `docs/philrvae_theory.tex` |
| `HyperbolicPhILRVAE` | `src/biomevae/models/hyperbolic_philrvae.py` | `docs/hyperbolic_philrvae_theory.md` | — |

## Coverage table — DIVA family

DIVA (Ilse et al., 2020) partitions the latent into three Gaussian factors
`z = [z_d ; z_y ; z_x]` with conditional priors `p(z_d|d)`, `p(z_y|y)`,
`p(z_x) = N(0, I)` and auxiliary classifiers on `z_d`, `z_y`.

| Model class | Backbone | Implementation file | Theory doc |
|---|---|---|---|
| `DIVABetaVAE` | β-VAE (Gaussian leaves) | `src/biomevae/models/diva_betavae.py` | `docs/diva_theory.md` |
| `DIVAHyperbolicPhILRVAE` | Hyperbolic PhILR | `src/biomevae/models/diva_hyp_philrvae.py` | `docs/diva_theory.md` + `docs/hyperbolic_philrvae_theory.md` |
| `DIVATreeDTMVAE` | TreeDTM | `src/biomevae/models/diva_treedtmvae.py` | `docs/diva_theory.md` + `docs/tree_dtm_vae_theory.md` |

The DIVA encoder / prior / classifier building blocks are defined in
`src/biomevae/models/diva.py` (backbone-agnostic). The per-backbone wrappers
supply only the reconstruction likelihood.

## Coverage table — PhyloDIVA family

PhyloDIVA augments DIVA with two domain-adaptation regularisers and a
phylogeny-aware smoothness term:

* a **gradient-reversed study critic** on `z_y` (DANN, Ganin & Lempitsky 2015)
  in `src/biomevae/models/phylo_da.py`,
* **Deep CORAL** (Sun & Saenko 2016) covariance matching on `z_x` per study,
* **phylogenetic smoothness** on the decoder's edge / contrast outputs in
  `src/biomevae/models/phylo_cov.py`.

| Model class | Backbone | Implementation file | Theory doc |
|---|---|---|---|
| `PhyloDIVABetaVAE` | β-VAE | `src/biomevae/models/phylodiva_betavae.py` | `docs/phylodiva_theory.md` |
| `PhyloDIVAHyperbolicPhILRVAE` | Hyperbolic PhILR | `src/biomevae/models/phylodiva_hyp_philrvae.py` | `docs/phylodiva_theory.md` + `docs/hyperbolic_philrvae_theory.md` |
| `PhyloDIVATreeDTMVAE` | TreeDTM | `src/biomevae/models/phylodiva_treedtmvae.py` | `docs/phylodiva_theory.md` + `docs/tree_dtm_vae_theory.md` |

## Coverage table — CAPDA-VAE family

CAPDA-VAE replaces *marginal* domain invariance with *conditional* (per-class)
alignment plus leak-free OOF stacking. The cross-cohort variant runs in the
LOSO pipelines; the single-study variant runs in the standard / meta pipelines
(conditional alignment dormant unless a within-study sub-cohort is supplied).

| Model class | Variant | Implementation file | Theory doc |
|---|---|---|---|
| `CAPDAVAE` | cross-cohort (LOSO) | `src/biomevae/models/capda_vae.py` (`capda_fit`) | `docs/capda_vae_theory.md` |
| `CAPDAVAE` | single-study | `src/biomevae/models/capda_vae.py` (`capda_fit_single_study`) | `docs/capda_vae_theory.md` |

## Cross-family theorems

| Topic | Theory doc |
|---|---|
| Taxonomic inductive bias vs. domain-invariant adaptation (formal trade-off between the PhILR / TreeDTM backbones and the DIVA / PhyloDIVA invariance constraints) | `docs/taxonomy_vs_domain_invariance_theorem.md` |
| CAPDA-VAE: conditional alignment + OOF stacking; cross-cohort and single-study variants | `docs/capda_vae_theory.md` |

## Shared utilities (not standalone models)

These modules are referenced by multiple theory docs:

| Module | Purpose | Referenced from |
|---|---|---|
| `src/biomevae/models/taxonomy_graph.py` | Sample-by-feature TaxonomyGraph backbone with rank-aware node aggregation | `philrvae_theory.md`, `tree_dtm_vae_theory.md` |
| `src/biomevae/models/taxonomy_tree.py` | Rooted-tree utilities, SBP construction, node aggregation | `philrvae_theory.md`, `tree_dtm_vae_theory.md` |
| `src/biomevae/models/tree_spec.py`, `philr_treespec.py` | Tree-spec dataclasses used by treeprior / PhILR pipelines | `treepriorvae_theory.md`, `philrvae_theory.md` |
| `src/biomevae/models/diva.py` | DIVA building blocks: encoder heads, conditional priors, auxiliary classifiers, `DIVALoss` | `diva_theory.md` |
| `src/biomevae/models/grl.py` | Gradient reversal layer (Ganin & Lempitsky 2015) | `phylodiva_theory.md` |
| `src/biomevae/models/phylo_da.py` | `LatentStudyCritic` + `coral_per_study` | `phylodiva_theory.md` |
| `src/biomevae/models/phylo_cov.py` | Tree-contrast smoothness penalties | `phylodiva_theory.md` |

## Verification command

```bash
python - <<'PY'
from pathlib import Path
docs = [
    ('VAE',                          'docs/vae_theory.md'),
    ('HyperbolicVAE',                'docs/hyperbolicvae_theory.md'),
    ('TaxonomyGraphVAE',             'docs/graphvae_theory.md'),
    ('TreeStructuredPriorVAE',       'docs/treepriorvae_theory.md'),
    ('DeepPhyloFusionVAE',           'docs/phylofusionvae_theory.md'),
    ('FlowXFormerVAE',               'docs/flowxformervae_theory.md'),
    ('HGVAE_ZI',                     'docs/hgvae_zi_theory.md'),
    ('TreeDTMVAE',                   'docs/tree_dtm_vae_theory.md'),
    ('PhILRVAE',                     'docs/philrvae_theory.md'),
    ('HyperbolicPhILRVAE',           'docs/hyperbolic_philrvae_theory.md'),
    ('DIVA (backbone-agnostic)',     'docs/diva_theory.md'),
    ('PhyloDIVA (regularisers)',     'docs/phylodiva_theory.md'),
    ('CAPDAVAE (cross + single)',    'docs/capda_vae_theory.md'),
]
for name, md in docs:
    print(f"{name:32s} md={Path(md).is_file()}")
PY
```

All entries should resolve to `md=True`.
