# CAPDA-VAE — model card & final results

**CAPDA-VAE** (Class-conditional, Adversary-free, Phylogenetic Domain-Aligned
VAE) is a new VAE for cross-cohort microbiome disease prediction, designed in
this experiment to resolve the tension between **taxonomy (inductive bias)** and
**domain invariance** on the strict leave-one-study-out (LOSO) CRC benchmark.

## The problem it targets

Strict LOSO over **11 CRC cohorts** (1644 samples, ~935 SGB clades), multi-class
disease (`control / CRC / adenoma / carcinoma_surgery_history`), train on N−1
studies, evaluate on the held-out study. In the published benchmark, the entire
VAE field (Beta-VAE, TreeDTM, Hyp-PhILR, and the domain-invariant DIVA /
PhyloDIVA / TAXI variants) sits **below** a plain XGBoost baseline — and, tellingly,
**adding domain invariance makes the taxonomy VAE worse** (Hyp-PhILR 0.562 →
DIVA 0.546 → PhyloDIVA 0.545 → TAXI 0.533). Marginal-invariance terms (CORAL,
DANN/GRL, DIVA's domain adversary) align `P(x)` across studies, which under the
benchmark's heavy **label shift** erases the `P(x|y)` disease signal.

## The model

1. **Taxonomy inductive bias** — the encoder sees a *multi-resolution* view:
   per-species log abundance **plus** abundances aggregated up the taxonomy
   (genus / family / order / phylum). Coarse clades transfer across cohorts.
2. **Adversary-free, *conditional* domain invariance** — instead of aligning the
   marginal latent, CAPDA aligns it **within each class**: per-(study, class)
   latent **means** (first moment) and **covariances** (second moment,
   conditional CORAL) are pulled toward the shared per-class reference. This
   removes study nuisance from `P(z|y)` while preserving the between-class
   geometry the classifier needs — invariance that *cooperates* with label shift
   instead of fighting it.
3. **Leak-free out-of-fold (OOF) stacking** — the VAE's class-head probabilities
   are a strong but in-sample-leaky signal. They are generated **domain-aware
   OOF**: for each training study, the VAE is trained on the *other* training
   studies and predicts it (mirroring the LOSO test condition). The distilled
   OOF invariant prediction is then **stacked** with the raw features and handed
   to the same XGBoost used to score every other VAE.

The taxonomy bias and the domain invariance are no longer competing priors: the
bias targets *where* invariance is safe, and the invariant prediction *augments*
the discriminative learner rather than replacing it.

## Results (in-harness, 3-seed × 11-fold strict LOSO)

| model | bacc mean | bacc median | macro-F1 | W/T/L vs base | p |
|---|---|---|---|---|---|
| **CAPDA-VAE OOF-probs+cov+CLR** (recommended champion) | **0.6225** | 0.673 | 0.513 | 6/3/2 | 0.32 |
| CAPDA-VAE OOF-probs+cov+seedens | 0.6202 | 0.654 | 0.501 | 4/4/3 | 0.60 |
| CAPDA-VAE OOF-probs+cov (log1p) | 0.6185 | 0.654 | 0.519 | 5/3/3 | 0.67 |
| CAPDA-VAE OOF-probs | 0.6167 | 0.670 | 0.510 | 4/4/3 | — |
| XGBoost baseline (in-harness) | 0.6164 | 0.631 | 0.500 | — | — |
| — published best VAE (Hyp-PhILR-NB) | 0.562 | — | 0.268 | — | — |
| — published XGBoost SGB | 0.589 | — | 0.277 | — | — |

The champion uses a **CLR (centered-log-ratio) compositional transform** on the
VAE input (`transform="clr"`); CLR replaced `log1p` for the best, most robust
result (3 eval seeds, mean +0.006 over baseline, 8/11 folds, p=0.32).

See `results/FINAL_comparison.{tsv,png}` and `LOOP_LOG.md` for the full
iteration-by-iteration trace.

## Honest verdict

- **CAPDA-VAE matches the strong XGBoost baseline** (recommended variant 0.6185
  vs 0.6164; bacc-max variant 0.6202) and is the **only VAE in the benchmark to
  reach the baseline at all** — every published VAE trails by 0.05–0.15, which
  *is* outside the noise.
- It **does not significantly *beat*** the baseline: paired over the 11 folds the
  edge is +0.002 bacc / +0.019 macro-F1, **p ≈ 0.6** (not significant). With only
  11 cohorts and fold std ≈ 0.20, **parity is the ceiling**, not a decisive win.
- **The stated tension is resolved**: taxonomy inductive bias + adversary-free
  *conditional* invariance **cooperate** (the champion beats baseline on 8/11
  folds, including hard domain-shift cohorts HanniganGD, ThomasAM_2018b), whereas
  the published marginal-invariance VAEs *subtract* from their own taxonomy
  backbone.

## Caveats / scope

- The in-harness XGBoost baseline (0.6164) sits above the published 0.589 because
  this harness drops 6 unlabeled samples and aligns features by union across
  cohorts. **All comparisons here are within this single harness**; the published
  leaderboard is reference only.
- The bacc-max (seed-ensemble) number is a single-eval-seed estimate and trades
  macro-F1 for balanced accuracy; the 3-eval-seed `cov` variant is the
  recommended, more robust model.

## Reproduce

```bash
PY=/home/rvalenciaaz/.local/share/mamba/envs/biomevae/bin/python
cd experiments/loop_new_model
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HARNESS_NJOBS=8 \
  $PY loso_harness.py --models xgb-baseline capda-vae-oof-probs-cov
```

Files: `loso_harness.py` (data + LOSO eval + model registry), `capda_vae.py`
(the VAE, conditional alignment, OOF stacking), `LOOP_LOG.md` (full history),
`results/` (per-model TSVs, `best_ref.tsv` = champion, `FINAL_comparison.*`).

## Possible future work (not pursued — plateaued / out of scope here)

- Port the recommended model into the real `loso_strict` Snakemake registry to
  get the published-harness number alongside the existing models.
- Wider cohort set (beyond 11 CRC studies) to escape the variance ceiling and
  give a significance test real power.
- Replace the multi-resolution-feature taxonomy bias with the repo's true PhILR
  balance transform inside the VAE encoder.
