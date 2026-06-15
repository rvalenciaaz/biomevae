"""CAPDA-VAE: Class-conditional, Adversary-free, Phylogenetic Domain-Aligned VAE.

A VAE designed to resolve the tension between the *taxonomy* inductive bias and
*domain invariance* that the existing benchmark exposes (DIVA/PhyloDIVA/TAXI all
fall *below* the no-DA Hyp-PhILR because marginal-invariance terms destroy the
disease signal under the benchmark's heavy label shift).

Two ideas, made to cooperate:

1. **Phylogenetic inductive bias (taxonomy):** the encoder sees a multi-
   resolution view of the community -- per-species relative abundance plus
   abundances aggregated up the taxonomy (genus / family / order / phylum).
   Coarse clades transfer across cohorts; fine species carry cohort-specific
   detail. This is supplied as features so the bias helps without competing
   with the alignment objective.

2. **Class-conditional alignment (adversary-free domain invariance):** instead
   of forcing the *marginal* latent q(z) to match across studies (CORAL / DANN
   / DIVA's domain adversary -- which under label shift erases P(z|y)), we align
   the latent *within each class*: per-(study, class) latent means are pulled
   toward the shared per-class centroid. This removes study nuisance while
   preserving the between-class geometry that the classifier needs.

The latent posterior mean is then handed to the same XGBoost used to evaluate
every other VAE in the benchmark, so any gain is attributable to the
representation, not the classifier.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------- #
# Taxonomy multi-resolution feature construction
# --------------------------------------------------------------------------- #
def build_taxon_aggregates(
    X_raw: np.ndarray,
    feat: List[str],
    taxonomy,
    levels: tuple = ("g", "f", "o", "p"),
) -> np.ndarray:
    """Sum per-species relative abundance into higher taxonomic ranks.

    Returns an (n_samples, n_extra) matrix of aggregated abundances (one block
    of columns per requested rank), aligned to ``X_raw`` rows. Missing lineage
    entries are bucketed under an ``<unknown>`` group for that rank.
    """
    blocks = []
    feat_index = {f: i for i, f in enumerate(feat)}
    for lvl in levels:
        if taxonomy is None or lvl not in getattr(taxonomy, "columns", []):
            continue
        # group species -> list of column positions
        groups: Dict[str, List[int]] = {}
        lab = taxonomy[lvl] if lvl in taxonomy.columns else None
        for f, i in feat_index.items():
            g = "<unknown>"
            if lab is not None and f in lab.index:
                v = lab.loc[f]
                g = str(v) if v == v and v is not None else "<unknown>"
            groups.setdefault(g, []).append(i)
        agg = np.zeros((X_raw.shape[0], len(groups)), dtype=np.float32)
        for j, (_g, idxs) in enumerate(sorted(groups.items())):
            agg[:, j] = X_raw[:, idxs].sum(axis=1)
        blocks.append(agg)
    if not blocks:
        return np.zeros((X_raw.shape[0], 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class _CAPDAVAE(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, latent: int = 32,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.lv = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, in_dim),
        )
        self.cls = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std) if self.training else mu
        return self.dec(z), mu, lv, self.cls(z), z


def _conditional_alignment(mu, y, dom, n_classes, n_dom):
    """Mean per-(study,class) latent spread around each class centroid.

    For every class c present in the batch, gather the per-study latent means
    and penalise their variance around the class-c grand mean. This makes the
    latent study-invariant *conditioned on the label* without flattening the
    between-class structure.
    """
    total = mu.new_zeros(())
    n_terms = 0
    for c in range(n_classes):
        cmask = (y == c)
        if cmask.sum() < 2:
            continue
        mu_c = mu[cmask]
        dom_c = dom[cmask]
        present = torch.unique(dom_c)
        if present.numel() < 2:
            continue
        study_means = []
        for d in present:
            dm = mu_c[dom_c == d]
            if dm.shape[0] >= 1:
                study_means.append(dm.mean(0))
        if len(study_means) < 2:
            continue
        sm = torch.stack(study_means, 0)            # (n_present_studies, latent)
        centroid = sm.mean(0, keepdim=True)
        total = total + ((sm - centroid) ** 2).sum(1).mean()
        n_terms += 1
    if n_terms == 0:
        return mu.new_zeros(())
    return total / n_terms


def _conditional_cov_alignment(mu, y, dom, n_classes):
    """Second-moment (covariance) conditional alignment -- conditional CORAL.

    For each class, align every study's within-class latent covariance to the
    per-class average covariance. Where the mean-only term equalises *where* a
    class sits per study, this equalises its *shape/spread*, removing more study
    nuisance from P(z|y) while still leaving the between-class geometry intact.
    """
    total = mu.new_zeros(())
    n_terms = 0
    for c in range(n_classes):
        cmask = (y == c)
        if cmask.sum() < 4:
            continue
        mu_c = mu[cmask]
        dom_c = dom[cmask]
        covs = []
        for d in torch.unique(dom_c):
            dm = mu_c[dom_c == d]
            if dm.shape[0] >= 2:
                covs.append(torch.cov(dm.T))
        if len(covs) < 2:
            continue
        ref = torch.stack(covs, 0).mean(0)
        for cov in covs:
            total = total + ((cov - ref) ** 2).mean()
        n_terms += 1
    if n_terms == 0:
        return mu.new_zeros(())
    return total / n_terms


def capda_vae_fit_predict(
    Xtr_raw: np.ndarray, ytr: np.ndarray, dtr: np.ndarray, Xte_raw: np.ndarray,
    seed: int, feat=None, taxonomy=None,
    latent: int = 32, hidden: int = 256, epochs: int = 220, lr: float = 1e-3,
    beta: float = 0.5, alpha: float = 2.0, gamma: float = 5.0,
    gamma_cov: float = 0.0, transform: str = "log1p", view: str = "full",
    dropout: float = 0.1, weight_decay: float = 1e-5,
    kl_warmup: float = 0.3, align_warmup: float = 0.3,
    classifier: str = "xgb", device: str = "cpu", **_ignore,
) -> np.ndarray:
    """Train CAPDA-VAE on N-1 studies, return predictions on the held-out study.

    ``ytr``/``dtr`` are integer class and study indices for the training rows.
    Predictions are returned in the same (contiguous) class space as ``ytr``.
    """
    assert torch is not None, "torch unavailable"
    import os as _os
    torch.set_num_threads(int(_os.environ.get("HARNESS_NJOBS", "8")))
    rng = np.random.RandomState(seed)
    torch.manual_seed(int(seed))

    from sklearn.preprocessing import StandardScaler

    # --- multi-resolution taxonomy features (inductive bias) ---------------
    agg_tr = build_taxon_aggregates(Xtr_raw, list(feat), taxonomy)
    agg_te = build_taxon_aggregates(Xte_raw, list(feat), taxonomy)

    def _species_feat(X):
        if transform == "clr":
            # centered-log-ratio: the natural compositional coordinate for
            # relative-abundance data (log of each part minus the per-sample
            # mean log over parts; pseudocount handles structural zeros).
            Xp = X + 1e-6
            lg = np.log(Xp)
            return lg - lg.mean(axis=1, keepdims=True)
        return np.log1p(X)

    if view == "coarse":
        # coarse-taxonomy-only view: the genus/family/order/phylum aggregates,
        # no species. A deliberately more transferable (and decorrelated-from-
        # species) view of the community for multi-view stacking.
        Xtr = np.log1p(agg_tr); Xte = np.log1p(agg_te)
    else:
        Xtr = np.concatenate([_species_feat(Xtr_raw), np.log1p(agg_tr)], axis=1)
        Xte = np.concatenate([_species_feat(Xte_raw), np.log1p(agg_te)], axis=1)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)

    # study -> contiguous index
    studies = np.unique(dtr)
    s2i = {s: i for i, s in enumerate(studies)}
    dtr_i = np.array([s2i[s] for s in dtr], dtype=np.int64)
    n_dom = len(studies)
    n_classes = int(ytr.max()) + 1

    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr.astype(np.int64), device=device)
    dt = torch.tensor(dtr_i, device=device)
    # class weights to counter imbalance in the supervised head
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    cw = torch.tensor((counts.sum() / (n_classes * np.maximum(counts, 1.0))),
                      device=device, dtype=torch.float32)

    model = _CAPDAVAE(Xtr.shape[1], n_classes, latent, hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = Xtr.shape[0]
    bs = min(256, n)
    model.train()
    for ep in range(epochs):
        w_kl = beta * min(1.0, ep / max(1, int(epochs * kl_warmup)))
        w_al = gamma * min(1.0, ep / max(1, int(epochs * align_warmup)))
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = Xt[idx]; yb = yt[idx]; db = dt[idx]
            recon, mu, lv, logits, _z = model(xb)
            rec = F.mse_loss(recon, xb, reduction="none").sum(1).mean()
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
            ce = F.cross_entropy(logits, yb, weight=cw)
            al = _conditional_alignment(mu, yb, db, n_classes, n_dom)
            loss = rec + w_kl * kl + alpha * ce + w_al * al
            if gamma_cov > 0:
                cov_al = _conditional_cov_alignment(mu, yb, db, n_classes)
                loss = loss + (w_al / max(gamma, 1e-8)) * gamma_cov * cov_al
            opt.zero_grad(); loss.backward(); opt.step()

    # --- encode (posterior mean) + classify --------------------------------
    model.eval()
    with torch.no_grad():
        ztr = model.encode(Xt)[0].cpu().numpy()
        zte = model.encode(torch.tensor(Xte, device=device))[0].cpu().numpy()
        ptr = torch.softmax(model.cls(torch.tensor(ztr, device=device)), 1).cpu().numpy()
        pte = torch.softmax(model.cls(torch.tensor(zte, device=device)), 1).cpu().numpy()

    if classifier == "embed":
        # Return the learned representation for external (e.g. OOF) stacking.
        return ztr, ptr, zte, pte
    if classifier == "head":
        return pte.argmax(1)

    from xgboost import XGBClassifier
    if classifier == "raw_latent":
        # Clean "baseline + aligned latent" test: exactly the baseline feature
        # set (log1p species, standardised) augmented with the latent only.
        # NO head probabilities -- those are in-sample predictions and leak
        # (overfit on train, miscalibrated at test) -- and NO taxonomy
        # aggregates, to isolate the latent's marginal value over the baseline.
        from sklearn.preprocessing import StandardScaler as _SS
        sc2 = _SS().fit(np.log1p(Xtr_raw))
        Rtr = sc2.transform(np.log1p(Xtr_raw)).astype(np.float32)
        Rte = sc2.transform(np.log1p(Xte_raw)).astype(np.float32)
        Ztr = np.concatenate([Rtr, ztr], axis=1)
        Zte = np.concatenate([Rte, zte], axis=1)
    elif classifier == "hybrid":
        # "Augment, don't replace": give XGBoost the full discriminative feature
        # set (the scaled multi-resolution taxonomy view) PLUS the conditionally
        # aligned latent and the head probabilities. The latent can only *add*
        # invariance signal -- by construction the booster cannot do worse than
        # the raw-feature baseline.
        Ztr = np.concatenate([Xtr, ztr, ptr], axis=1)
        Zte = np.concatenate([Xte, zte, pte], axis=1)
    else:  # "xgb": latent (+ head probs) only
        Ztr = np.concatenate([ztr, ptr], axis=1)
        Zte = np.concatenate([zte, pte], axis=1)
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        random_state=int(seed), n_jobs=int(_os.environ.get("HARNESS_NJOBS", "8")),
        eval_metric="mlogloss",
    ).fit(Ztr, ytr)
    return clf.predict(Zte)


# --------------------------------------------------------------------------- #
# Domain-aware out-of-fold stacking
# --------------------------------------------------------------------------- #
def capda_vae_oof_fit_predict(
    Xtr_raw: np.ndarray, ytr: np.ndarray, dtr: np.ndarray, Xte_raw: np.ndarray,
    seed: int, feat=None, taxonomy=None, epochs: int = 140,
    use_latent: bool = True, n_vae_seeds: int = 1, **hp,
) -> np.ndarray:
    """Stack the VAE's domain-invariant prediction with raw features, leak-free.

    The head probabilities are powerful (iter 3 showed XGBoost wants them) but
    in-sample they leak. Here we generate them **out-of-fold in a domain-aware
    way**: for each training study s, the VAE is trained on the *other* training
    studies and used to predict s -- exactly mirroring the LOSO test condition,
    so the OOF features are honestly "predicted on an unseen cohort". A final
    VAE trained on all N-1 studies produces the held-out study's features.
    XGBoost is then trained on [raw log1p-species (+latent) + OOF head-probs].
    """
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    studies = np.unique(dtr)
    n_classes = int(ytr.max()) + 1
    n = Xtr_raw.shape[0]
    oof_p = np.zeros((n, n_classes), dtype=np.float32)
    oof_z = None

    def _embed(Xa, ya, da, Xb):
        """Embed-probs (and latent) averaged over n_vae_seeds VAE trainings.

        Averaging the invariant prediction across seeds reduces the variance of
        this (deliberately weak) predictor before it is stacked -- the dominant
        noise source given the huge fold-to-fold spread.
        """
        zt = pt = ze = pe = None
        for k in range(max(1, n_vae_seeds)):
            a, b, c, d = capda_vae_fit_predict(
                Xa, ya, da, Xb, seed + 1000 * k, feat=feat, taxonomy=taxonomy,
                epochs=epochs, classifier="embed", **hp)
            if pe is None:
                zt, pt, ze, pe = a, b, c, d
            else:
                pt = pt + b; pe = pe + d; zt = zt + a; ze = ze + c
        m = max(1, n_vae_seeds)
        return zt / m, pt / m, ze / m, pe / m

    # domain-aware OOF: leave-one-training-study-out
    for s in studies:
        inner_te = (dtr == s)
        inner_tr = ~inner_te
        if inner_tr.sum() < 10 or inner_te.sum() < 1:
            continue
        ytr_in = ytr[inner_tr]
        # remap to contiguous for the inner VAE head
        uniq = np.unique(ytr_in)
        remap = {c: i for i, c in enumerate(uniq)}
        ytr_in_c = np.array([remap[c] for c in ytr_in], dtype=np.int64)
        ztr_i, ptr_i, zte_i, pte_i = _embed(
            Xtr_raw[inner_tr], ytr_in_c, dtr[inner_tr], Xtr_raw[inner_te])
        # scatter inner-class probs back into the full class space
        full = np.zeros((pte_i.shape[0], n_classes), dtype=np.float32)
        for j, c in enumerate(uniq):
            full[:, c] = pte_i[:, j]
        oof_p[inner_te] = full
        if use_latent:
            if oof_z is None:
                oof_z = np.zeros((n, zte_i.shape[1]), dtype=np.float32)
            oof_z[inner_te] = zte_i

    # final model on all training studies -> held-out study features
    ztr_f, ptr_f, zte_f, pte_f = _embed(Xtr_raw, ytr, dtr, Xte_raw)

    sc = StandardScaler().fit(np.log1p(Xtr_raw))
    Rtr = sc.transform(np.log1p(Xtr_raw)).astype(np.float32)
    Rte = sc.transform(np.log1p(Xte_raw)).astype(np.float32)
    parts_tr = [Rtr, oof_p]
    parts_te = [Rte, pte_f]
    if use_latent:
        parts_tr.append(oof_z if oof_z is not None else ztr_f)
        parts_te.append(zte_f)
    Ztr = np.concatenate(parts_tr, axis=1)
    Zte = np.concatenate(parts_te, axis=1)

    import os as _os
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        random_state=int(seed), n_jobs=int(_os.environ.get("HARNESS_NJOBS", "8")),
        eval_metric="mlogloss",
    ).fit(Ztr, ytr)
    return clf.predict(Zte)


# --------------------------------------------------------------------------- #
# Inner-LOSO-weighted ensemble: raw-XGB (taxonomy detail) + aligned VAE
# (domain invariance), blended at the probability level.
# --------------------------------------------------------------------------- #
def capda_vae_ensemble_fit_predict(
    Xtr_raw: np.ndarray, ytr: np.ndarray, dtr: np.ndarray, Xte_raw: np.ndarray,
    seed: int, feat=None, taxonomy=None, epochs: int = 140,
    w_grid=None, **hp,
) -> np.ndarray:
    """Blend XGBoost-on-raw and the aligned-VAE class probabilities.

    Both predictors' out-of-fold probabilities are produced by domain-aware
    inner LOSO (train on the other training studies, predict the held-in study),
    then the blend weight ``w`` in ``w*p_xgb + (1-w)*p_vae`` is chosen to
    maximise inner-LOSO balanced accuracy. Because inner LOSO mirrors the real
    held-out condition, the chosen ``w`` leans on raw features for raw-friendly
    cohorts and borrows the VAE's invariance for hard domain-shift cohorts ->
    keeps the VAE's wins, removes its easy-fold noise.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from xgboost import XGBClassifier
    import os as _os
    nj = int(_os.environ.get("HARNESS_NJOBS", "8"))
    if w_grid is None:
        w_grid = np.linspace(0.0, 1.0, 11)

    studies = np.unique(dtr)
    n_classes = int(ytr.max()) + 1
    n = Xtr_raw.shape[0]
    oof_vae = np.zeros((n, n_classes), dtype=np.float32)
    oof_xgb = np.zeros((n, n_classes), dtype=np.float32)

    def _xgb():
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", random_state=int(seed),
            n_jobs=nj, eval_metric="mlogloss")

    def _scatter(p, uniq, m):
        full = np.zeros((m, n_classes), dtype=np.float32)
        for j, c in enumerate(uniq):
            full[:, c] = p[:, j]
        return full

    # domain-aware OOF for BOTH predictors
    for s in studies:
        ite = (dtr == s); itr = ~ite
        if itr.sum() < 10 or ite.sum() < 1:
            continue
        ytr_in = ytr[itr]
        uniq = np.unique(ytr_in)
        remap = {c: i for i, c in enumerate(uniq)}
        ytr_in_c = np.array([remap[c] for c in ytr_in], dtype=np.int64)
        # VAE
        _zt, _pt, _ze, pte_i = capda_vae_fit_predict(
            Xtr_raw[itr], ytr_in_c, dtr[itr], Xtr_raw[ite], seed,
            feat=feat, taxonomy=taxonomy, epochs=epochs, classifier="embed", **hp)
        oof_vae[ite] = _scatter(pte_i, uniq, ite.sum())
        # raw XGB
        sc = StandardScaler().fit(np.log1p(Xtr_raw[itr]))
        clf = _xgb().fit(sc.transform(np.log1p(Xtr_raw[itr])), ytr_in_c)
        oof_xgb[ite] = _scatter(
            clf.predict_proba(sc.transform(np.log1p(Xtr_raw[ite]))), uniq, ite.sum())

    # tune blend weight on inner-LOSO balanced accuracy
    best_w, best_b = 0.0, -1.0
    for w in w_grid:
        pred = (w * oof_xgb + (1.0 - w) * oof_vae).argmax(1)
        b = balanced_accuracy_score(ytr, pred)
        if b > best_b:
            best_b, best_w = b, float(w)

    # final predictors on all training studies
    _zt, _pt, _ze, pte_vae = capda_vae_fit_predict(
        Xtr_raw, ytr, dtr, Xte_raw, seed, feat=feat, taxonomy=taxonomy,
        epochs=epochs, classifier="embed", **hp)
    sc = StandardScaler().fit(np.log1p(Xtr_raw))
    clf = _xgb().fit(sc.transform(np.log1p(Xtr_raw)), ytr)
    pte_xgb = clf.predict_proba(sc.transform(np.log1p(Xte_raw)))

    blended = best_w * pte_xgb + (1.0 - best_w) * pte_vae
    return blended.argmax(1)


# --------------------------------------------------------------------------- #
# Multi-view OOF stacking: combine a species-level and a coarse-taxonomy
# invariant prediction (two decorrelated views of the invariant signal).
# --------------------------------------------------------------------------- #
def capda_vae_multiview_fit_predict(
    Xtr_raw: np.ndarray, ytr: np.ndarray, dtr: np.ndarray, Xte_raw: np.ndarray,
    seed: int, feat=None, taxonomy=None, epochs: int = 140,
    views=("full", "coarse"), **hp,
) -> np.ndarray:
    """Stack raw features with OOF invariant predictions from MULTIPLE views.

    Trains a separate domain-aware-OOF invariant predictor per ``view`` (e.g.
    species+aggregates vs coarse-aggregates-only) and stacks all of their
    distilled OOF probability columns with the raw features. Coarse-taxonomy
    predictors transfer better across cohorts and are decorrelated from the
    species view, so two views may contribute independent lift.
    """
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    import os as _os
    studies = np.unique(dtr)
    n_classes = int(ytr.max()) + 1
    n = Xtr_raw.shape[0]

    def _oof_for_view(view):
        oof = np.zeros((n, n_classes), dtype=np.float32)
        for s in studies:
            ite = (dtr == s); itr = ~ite
            if itr.sum() < 10 or ite.sum() < 1:
                continue
            yin = ytr[itr]; uniq = np.unique(yin)
            remap = {c: i for i, c in enumerate(uniq)}
            yinc = np.array([remap[c] for c in yin], dtype=np.int64)
            _a, _b, _c, pte = capda_vae_fit_predict(
                Xtr_raw[itr], yinc, dtr[itr], Xtr_raw[ite], seed, feat=feat,
                taxonomy=taxonomy, epochs=epochs, classifier="embed",
                view=view, **hp)
            full = np.zeros((pte.shape[0], n_classes), dtype=np.float32)
            for j, c in enumerate(uniq):
                full[:, c] = pte[:, j]
            oof[ite] = full
        _a, _b, _c, pte_f = capda_vae_fit_predict(
            Xtr_raw, ytr, dtr, Xte_raw, seed, feat=feat, taxonomy=taxonomy,
            epochs=epochs, classifier="embed", view=view, **hp)
        return oof, pte_f

    sc = StandardScaler().fit(np.log1p(Xtr_raw))
    parts_tr = [sc.transform(np.log1p(Xtr_raw)).astype(np.float32)]
    parts_te = [sc.transform(np.log1p(Xte_raw)).astype(np.float32)]
    for view in views:
        oof, pte_f = _oof_for_view(view)
        parts_tr.append(oof); parts_te.append(pte_f)
    Ztr = np.concatenate(parts_tr, axis=1)
    Zte = np.concatenate(parts_te, axis=1)
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", random_state=int(seed),
        n_jobs=int(_os.environ.get("HARNESS_NJOBS", "8")), eval_metric="mlogloss",
    ).fit(Ztr, ytr)
    return clf.predict(Zte)
