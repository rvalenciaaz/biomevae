# Repository Audit Report — `biomevae`

**Date:** 2026-04-07 (reverified same day)
**Scope:** 65+ files, ~16,400 lines of Python, plus shell scripts, configs, and environment files.
**Method:** Static analysis only (no files executed or modified).

Each finding below has been verified by reading the relevant source lines.
Items from the initial audit that turned out to be false positives are listed
at the end with brief explanations.

---

## Critical / High Severity (3 confirmed)

### H1. `SystemExit` raised in library code — `flowxformer.py:72-75`

```python
if branchlen_mode not in {"unit", "rank"}:
    raise SystemExit("branchlen_mode must be 'unit' or 'rank'.")
if not feature_clades:
    raise SystemExit("feature_clades must contain at least one entry.")
```

`SystemExit` terminates the entire process. Library/model code should raise
`ValueError`. This kills any calling application (e.g., a Jupyter notebook or
test harness).

### H2. Missing `scipy` dependency — `pyproject.toml`

`scipy` is used in `reconstruction.py:27` (`from scipy.optimize import nnls`)
and in `benchmark_figures_enterosignatures.py:11` and `interpret_compare.py:145`,
but is not listed in `[project.dependencies]`. Runtime crash on
`cross_validate_nmf` and related functions.

### H3. `train_loop.py` passes wrong phylo weights for `phylo_fusion` at eval — `train_loop.py:366-370`

```python
model.eval()
with torch.no_grad():
    all_tensor = torch.from_numpy(X_proc).to(device)   # X_proc is standardised
    mu, logvar = model.encode(all_tensor)               # phylo_weights=None → uses X_proc
```

During training (line 174-175), `model(xb, phylo_weights=phylo_weights)` passes
**raw unstandardised** counts for the phylo summary. At eval time,
`model.encode(all_tensor)` falls back to `phylo_weights=None`, which uses
standardised data (can be negative) for the phylo summary. The
`_phylo_summary()` method clamps to `min=0`, so negative standardised values
become 0, producing a different phylo representation than what the model trained
on.

**Verified:** `DeepPhyloFusionVAE.encode()` at `phylo_fusion.py:66-67` confirms
`weights = x if phylo_weights is None else phylo_weights`.

---

## Medium Severity (11 confirmed)

### M1. Wrong clustering parameter — `benchmark_figures_enterosignatures.py:1238`

```python
vae_labels = cluster_embeddings(latent, enterosig.n_components, ...)
```

Should be `enterosig.n_clusters` (consistent with line 1038-1040 in the same
file). The `cluster_embeddings()` function parameter is `n_clusters: int`
(`enterosignatures.py:539`). `n_components` is the NMF rank; `n_clusters` is the
k-medoids count — these can differ.

### M2. `_select_alpha` picks smallest alpha — `enterosignatures.py:394-399`

```python
selected_alpha = float(alpha_values[-1])   # fallback: LARGEST alpha
for alpha in alpha_values:                  # iterate smallest → largest
    if mean_ev[float(alpha)] >= threshold:
        selected_alpha = float(alpha)
        break                               # picks SMALLEST meeting threshold
```

The fallback to the largest alpha and the sparsity goal suggest the intent was
the **largest** alpha that still meets the EV threshold. The loop picks the
smallest instead.

### M3. `val_loss`/`val_recon`/`val_kld` undefined if `epochs=0` — `train_loop.py:386-390`

The loop at line 157 (`for epoch in range(1, params["epochs"] + 1)`) never
executes when `epochs=0`, but line 388 references `val_loss`. No validation
guards against `epochs <= 0`.

### M4. Per-dimension KL diagnostics ignore conditional priors — `train_loop.py:285-287`

```python
kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
```

This computes standard-normal KL, but the actual training loss at line 250-256
uses `kl_per_sample(... prior_mu=prior_mu, prior_logvar=prior_logvar)`. For
tree-prior models the per-dimension diagnostic numbers are wrong.

### M5. Optuna config vs. hardcoded range collision — `optuna_utils.py:214-218`

In `build_trial_params()`, `suggest_params()` calls e.g.
`trial.suggest_float("lr", 1e-4, 5e-3, log=True)` (line 63). Then
`_suggest_from_config()` may call `trial.suggest_float("lr", low=..., high=...)`
with different bounds from the JSON config. Optuna raises `ValueError` when the
same parameter is suggested with different ranges.

### M6. `--verbose` always `True` — `all_methods.py:68-71`

```python
parser.add_argument("--verbose", action="store_true", default=True)
```

`default=True` with `action="store_true"` means the flag is always True with no
way to disable it. Compare `benchmark_figure.py` which provides a
`--verbose`/`--no-verbose` pair via `set_defaults`.

### M7. `--taxonomy` passed twice — `hpc/postprocess_model.sh:186-192`

```bash
local CMD=("${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-interpret
    ...
    --taxonomy "${TAXONOMY}"     # ← explicit
    --taxonomy-level genus
    "${tax_args[@]}"             # ← already contains --taxonomy "${TAXONOMY}" (line 70)
)
```

`tax_args` is set at line 68-71: `tax_args=(--taxonomy "${TAXONOMY}")`. The
function `run_interpret_genus` (only reached when `TAXONOMY != "none"`) passes
`--taxonomy` twice. Other functions (`run_test`, `run_embed`, `run_interpret`)
correctly use only `"${tax_args[@]}"`.

### M8. `ddof=1` with single fold produces `nan` — `reconstruction.py:89`

```python
std[key] = float(values.std(ddof=1))
```

With a single fold, `values` has 1 element and `std(ddof=1)` divides by 0,
yielding `nan` that silently propagates into `std_metrics`.

### M9. `uot_lambda` stored but never used — `flowxformer.py:203`

`self.uot_lambda = float(uot_lambda)` is set in `FlowFeaturizer.__init__` and
passed from `FlowXFormerVAE` (line 421) but never referenced in any method of
`FlowFeaturizer` (`_leaf_mass`, `compute_flows`, `forward`). Dead/incomplete
feature.

### M10. `rank_vocab` parameter dead in `TreeNBVAE` — `treenbvae.py:310`

```python
def __init__(self, hidden: int, latent_dim: int, rank_vocab: int, topo, ...):
```

`rank_vocab` is accepted but never passed to `TreeNBEncoder` or
`TreeSoftmaxDecoder`, or used anywhere in the body. Dead parameter.

### M11. `_explained_variance` is not R-squared — `enterosignatures.py:168-173`

```python
denom = float(np.sum(observed ** 2))            # ← sum(x²)
```

Standard R² uses `sum((observed - mean)²)` as the denominator. The function
computes "fraction of energy explained" but is named `_explained_variance`,
which is misleading.

---

## Low Severity (21 confirmed)

### L1. `FlowXFormerVAE` not exported from `models/__init__.py`

The only model class with a dedicated CLI entry point that is absent from the
package's `__init__.py`. `from biomevae.models import FlowXFormerVAE` raises
`AttributeError`.

### L2. `cli/__init__.py` only exposes 14 of 27 entry-point modules

13 CLI modules (visualization/analysis scripts: `nmf_baseline`,
`compare_to_nmf`, `all_methods`, `benchmark_figure`, `benchmark_slides`,
`benchmark_figures_enterosignatures`, `vae_interpret`, `interpret_compare`,
`plot_training_curves`, `recon_scatter`, `recon_violin`, `hierarchy_figure`,
`pairwise_table`) are not in `__all__`/`_lazy_modules`. Programmatic access via
`biomevae.cli.<name>` raises `AttributeError`.

### L3. Inconsistent `forward()` return types across models

- Tuple `(recon, mu, logvar)`: `VAE`, `TaxonomyGraphVAE`, `FlowXFormerVAE`,
  `DeepPhyloFusionVAE`, `TreeStructuredPriorVAE`, `HyperbolicVAE`, `PhILRVAE`
- Dict: `HGVAE_ZI`, `TreeNBVAE`

### L4. Inconsistent `reparameterize` naming

- `reparameterize` (static): `VAE`, `TaxonomyGraphVAE`, `FlowXFormerVAE`,
  `DeepPhyloFusionVAE`, `TreeStructuredPriorVAE`
- `reparam` (static): `HGVAE_ZI`, `TreeNBVAE`, `PhILRVAE`
- `_reparameterize_hyperbolic` (instance, private): `HyperbolicVAE`

### L5. Massive code duplication across CLI scripts

- `_load_results`, `_parse_figsize`, `_parse_renames`: duplicated across
  `benchmark_figure.py`, `hierarchy_figure.py`, `pairwise_table.py`,
  `recon_violin.py`
- `_load_config`, `_load_scaler`, `_apply_scaler`, `_build_model`: ~130-line
  blocks duplicated across `vae_embed.py`, `vae_test.py`, `vae_interpret.py`

### L6. `parse_int_list` mishandles negatives — `_recon_cli.py:49`

Splitting on `-` to detect ranges means `"-5"` splits into `["", "5"]`, failing
with `int("")`. Negative integers are treated as invalid ranges. In practice
this function is only used for positive rank candidates so it doesn't trigger.

### L7. Error handling inconsistencies across CLIs

- `plot_training_curves.py` raises `ValueError`/`FileNotFoundError` (tracebacks)
- `vae_train_hgvae_zi.py` raises `ValueError` for CLI validation (line 535)
- All other CLIs raise `SystemExit` (clean error messages)

### L8. `plot_training_curves.py` — `main()` takes no `argv` parameter

Unlike all other CLI scripts, `main()` (line 106) and `parse_args()` (line 14)
accept no `argv`, making programmatic testing impossible.

### L9. Literal tab vs `"\t"` inconsistency

`vae_embed.py` and `vae_test.py` use literal tab characters as TSV separators
in hgvae_zi code paths; all other paths use escaped `"\t"`. Fragile to editor
corruption.

### L10. Dense adjacency for taxonomy graphs — `taxonomy_graph.py:11-26`

`_build_normalized_adjacency` creates a full `(num_nodes, num_nodes)` dense
matrix. For 50k+ nodes this requires ~10GB RAM.

### L11. `taxonomy.py` Laplacian hardcoded to 3 levels — `taxonomy.py:109-127`

```python
ws = lap_w[0]; wg = lap_w[1]; wf = lap_w[2]
add_level("s", ws); add_level("g", wg); add_level("f", wf)
```

Always uses species/genus/family regardless of what `levels` parameter was
passed.

### L12. `data_testing/ParseMeta.py` — 10+ unused imports, dead debugger

`numpy`, `os`, `subprocess`, `re`, `RandomState`, `logging`, `PIPE`,
`defaultdict`, `Counter`, `date` are all imported but never used. Contains
`#import ipdb; ipdb.set_trace()`.

### L13. `requirement_simple.yml` is non-functional

Missing `torch`, `scikit-learn`, `scipy`, and the package itself.

### L14. `requirements.yml` missing `scikit-learn`

Hard dependency in `pyproject.toml` but absent from conda environment file.

### L15. `requirements.yml` treats optional deps as mandatory

`optuna` and `geoopt` are listed as hard requirements but are optional in
`pyproject.toml`.

### L16. `PhILRVAE` import not guarded — `models/__init__.py:43`

Unconditional import triggers transitive chain through `flowxformer` →
`taxonomy`, defeating lazy-loading architecture. Other optional-dep models use
try/except guards.

### L17. Dead stored attributes

- `graph_mode`, `gnn_type` in `graph.py:38-39` and `treeprior.py:31-32`
- `phylo_method`, `phylo_dim_requested` in `phylo_fusion.py:29-30`

### L18. Unused imports in `vae_train_philrvae.py:17,23`

`Optional`, `Tuple` (from `typing`), and `save_scaler` (from `biomevae.data`)
imported but never used anywhere in the file.

### L19. `shutil` imported but unused — `tests/test_all_models.py:11`

### L20. `sys.path` manipulation in tests — `test_flowxformer.py:9`, `test_gabriel_column_masking.py:17`

Fragile; should rely on editable install or `conftest.py`.

### L21. Test swallows all exceptions — `test_gabriel_column_masking.py:131`

`except Exception: pass` could mask real bugs in `cross_validate_vae`.

---

## Design Concerns (not bugs, but worth noting)

### D1. `select_nmf_rank` always minimizes the selection metric — `reconstruction.py:524`

```python
best_rank = min(unique_candidates, key=lambda n: float(results[n].mean_metrics[selection_metric]))
```

The docstring explicitly says "minimizing" and the default metric is `"rmse"`
(lower is better), which is correct. No caller in the codebase currently passes
a higher-is-better metric. However, there is no direction parameter or guard, so
a caller using `selection_metric="r2"` would silently get wrong results.

### D2. `HyperbolicVAE` decoder receives Poincare ball points via Euclidean layers — `hyperbolic.py:68-71`

The decoder applies standard `nn.Linear` layers to latent `z` that lives on the
Poincare ball. A "pure" hyperbolic approach would use `logmap0` to project back
to tangent space first. The KL is computed in tangent space (via
`kl_per_sample`), which is valid for the wrapped-normal parameterization. This is
a common practical simplification in hyperbolic VAE implementations, not a
clear-cut mathematical error.

### D3. `TreeStructuredPriorVAE.forward()` does not invoke the tree prior — `treeprior.py:125-129`

`forward()` returns `(recon, mu, logvar)` without calling `conditional_prior()`.
However, the training loop at `train_loop.py:180-183` **correctly** calls
`conditional_prior()` and integrates the prior into the loss. The model works
correctly in its intended context; the concern is only about standalone usage.

### D4. Inconsistent logvar clamping across models

`kl_per_sample()` in `losses.py:66` already clamps logvar to `[-30, 20]`,
protecting the KL computation. Some models additionally clamp in `encode()`:
`FlowXFormerVAE` at `[-20, 20]`, `PhILRVAE`/`TreeNBVAE` at `[-10, 10]`. Other
models (`VAE`, `HyperbolicVAE`, `TaxonomyGraphVAE`, `DeepPhyloFusionVAE`,
`TreeStructuredPriorVAE`) do not clamp in encode, relying solely on the loss
function's clamp. The risk during reparameterization (`exp(0.5 * logvar)`) is
low since logvar would need to exceed ~176 for float32 overflow.

### D5. `clamp_min(0.0)` on `exp()` output is a no-op — `hgvae_zi.py:271`

```python
mean_pos = torch.exp(mu_log + 0.5 * sig * sig).clamp_min(0.0)
```

`torch.exp()` always returns non-negative values; the clamp is redundant.

### D6. Placeholder author in `pyproject.toml:12`

`authors = [{ name = "You", email = "you@example.com" }]`

---

## Retracted Findings (false positives from initial audit)

| Original ID | Claim | Why retracted |
|---|---|---|
| H1 | `free_bits` semantics inverted in `losses.py:75` | `torch.clamp(per_dim, min=free_bits)` IS the standard free-bits formulation (Kingma et al. 2016). Dimensions below the threshold have zero gradient, preventing posterior collapse. The implementation is correct. |
| M5 | `model.decoder(mu)` incompatible for some models | All models routed through `train_loop.py` (`VAE`, `HyperbolicVAE`, `TaxonomyGraphVAE`, `TreeStructuredPriorVAE`, `DeepPhyloFusionVAE`) have `self.decoder = nn.Sequential(...)`, which accepts a single tensor. Verified in `graph.py:73-74`, `treeprior.py:67`, `phylo_fusion.py:56-57`. |
| M9 | PhILR contrast matrix dimension wrong for non-binary trees | The SBP decomposition always produces exactly `p-1` contrasts regardless of tree structure: `sum over internal nodes of (children_i - 1) = (total edges from ≥2-child nodes) - (count of ≥2-child nodes) = p - 1`. Verified mathematically. |
| M11 | DataFrame mutation corrupts source in `enterosignatures.py:103` | `abundances` is always a distinct DataFrame: either via `.apply().fillna()` (line 87) or `.loc[bool_mask]` reassignment (line 96). Setting a column on it does not corrupt the original `raw` DataFrame. |
| M16 | Embedding index dtype crash | PyTorch's `nn.Embedding` internally casts `int32` indices to `int64`. While not matching the documented API contract (`torch.long`), this does not crash in any current PyTorch version. Downgraded to a style note. |

---

## Summary

| Severity | Count | Key Themes |
|----------|-------|------------|
| **Critical/High** | 3 | `SystemExit` in library code, missing `scipy` dependency, wrong phylo weights at eval |
| **Medium** | 11 | Wrong clustering parameter, alpha selection logic, undefined variables, KL diagnostic mismatch, Optuna range collision, duplicate CLI flags, dead parameters |
| **Low** | 21 | Dead code/imports, duplication, naming inconsistencies, missing exports, non-functional env files, fragile test patterns |
| **Design concerns** | 6 | Metric direction, hyperbolic decoder simplification, logvar clamp inconsistency, no-op clamp, placeholder metadata |
| **Retracted** | 5 | `free_bits` (correct), `model.decoder` (works), PhILR matrix (correct math), DataFrame mutation (safe), embedding dtype (works) |
| **Total confirmed** | **35** | + 6 design notes |
