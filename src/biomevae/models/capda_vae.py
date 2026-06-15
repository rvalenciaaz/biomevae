"""CAPDA-VAE: Class-conditional, Adversary-free, Phylogenetic Domain-Aligned VAE.

A VAE for cross-cohort (strict-LOSO) microbiome disease prediction that resolves
the tension between the *taxonomy* inductive bias and *domain invariance*: the
published DA-VAEs (DIVA / PhyloDIVA / TAXI) fall *below* their own no-DA
taxonomy backbone because they align the *marginal* ``q(z)`` across studies,
which under label shift erases the ``P(z|y)`` disease signal.  CAPDA instead

1. **Taxonomy inductive bias** — the encoder sees a multi-resolution view:
   per-species abundance in **CLR** (centred-log-ratio) compositional
   coordinates *plus* abundances aggregated up the taxonomy
   (genus / family / order / phylum).
2. **Adversary-free, *conditional* domain invariance** — the latent is aligned
   *within each class*: per-(study, class) latent **means** and **covariances**
   (conditional CORAL) are pulled toward the shared per-class reference, removing
   study nuisance from ``P(z|y)`` while preserving the between-class geometry.
3. **Leak-free domain-aware OOF stacking** — the VAE's class-head probabilities
   are powerful but in-sample-leaky, so they are produced **out-of-fold in a
   domain-aware way** (for each training study, the VAE is trained on the *other*
   training studies and predicts it, mirroring the LOSO test condition).  The
   distilled OOF probabilities are stacked with the raw ``log1p`` features and
   handed to the same XGBoost the rest of the LOSO sweep uses.

Pipeline integration (see ``cli/vae_train_capda_vae.py`` and
``cli/loso_strict_encode.py``): the trainer writes an ``embeddings.tsv`` of
``[log1p-species | OOF invariant-prob columns]`` for the training samples and
saves the final VAE; the strict encode step applies that final VAE to the
held-out cohort to produce the same columns (final-VAE probs).  ``biomevae-
loso-classify`` then trains XGBoost on the train rows and evaluates the held-out
rows exactly as for every other model — so the numbers are directly comparable.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # torch is a hard dependency of the trainers, optional for import-time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None


# Champion hyper-parameters (``capda-vae-oof-probs-cov-clr`` — see
# experiments/loop_new_model/MODEL_CARD.md).  Exposed so the CLI can surface
# them as flags while keeping the validated defaults.
DEFAULTS = dict(
    latent=32, hidden=256, epochs=140, lr=1e-3,
    beta=0.5, alpha=2.0, gamma=5.0, gamma_cov=2.0,
    dropout=0.1, weight_decay=1e-5, kl_warmup=0.3, align_warmup=0.3,
    transform="clr", agg_levels=("g", "f", "o", "p"),
)
PROB_COL_PREFIX = "capda_prob_"
FEAT_COL_PREFIX = "feat_"


# --------------------------------------------------------------------------- #
# Taxonomy multi-resolution features
# --------------------------------------------------------------------------- #
_TAX_LEVELS = ["k", "p", "c", "o", "f", "g", "s"]


def load_lineage_table(path: str, has_header: bool = False):
    """Load a clade -> lineage table indexed by clade (columns ``k..s``).

    The LOSO pipeline writes ``phyla.tsv`` **header-less** with the canonical
    8-column ``[clade, k, p, c, o, f, g, s]`` schema (see
    :func:`biomevae.loso.MergedStudies.write`).  ``biomevae.taxonomy.
    load_taxonomy_table`` assumes a header and mis-parses that file (the lineage
    columns collapse to ``NA_*``), so for the header-less default we parse the
    fixed-position columns directly — matching ``build_taxonomy_graph_from_
    phyla_tsv(has_header=False)`` that the other taxonomy models use.  When a
    real header is present (``--taxonomy-has-header``) we defer to the
    name-aware ``load_taxonomy_table``.
    """
    import pandas as pd
    if has_header:
        from biomevae.taxonomy import load_taxonomy_table
        return load_taxonomy_table(path)
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    ncol = df.shape[1]
    names = ["clade"] + _TAX_LEVELS
    if ncol >= len(names):
        df = df.iloc[:, :len(names)].copy()
        df.columns = names
    else:
        df = df.copy()
        df.columns = names[:ncol]
        for lv in _TAX_LEVELS:
            if lv not in df.columns:
                df[lv] = f"NA_{lv}"
    return df.drop_duplicates(subset="clade").set_index("clade")


def build_taxon_aggregates(
    X_raw: np.ndarray, feat_clades: List[str], taxonomy,
    levels: Tuple[str, ...] = ("g", "f", "o", "p"),
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Sum per-species abundance into higher taxonomic ranks.

    ``taxonomy`` is the table from :func:`biomevae.taxonomy.load_taxonomy_table`
    (indexed by clade, columns ``k p c o f g s``).  Returns ``(agg, groups)``
    where ``agg`` is ``(n_samples, n_groups)`` and ``groups`` is the ordered
    list of ``(level, group_name)`` keys — returned so the encode step can
    reproduce the exact same column order deterministically.
    """
    blocks: List[np.ndarray] = []
    group_keys: List[Tuple[str, str]] = []
    tax_cols = set(getattr(taxonomy, "columns", []))
    for lvl in levels:
        if taxonomy is None or lvl not in tax_cols:
            continue
        lab = taxonomy[lvl]
        groups: Dict[str, List[int]] = {}
        for i, clade in enumerate(feat_clades):
            g = "<unknown>"
            if clade in lab.index:
                v = lab.loc[clade]
                # Defend against duplicate clade rows -> Series
                if hasattr(v, "iloc"):
                    v = v.iloc[0]
                g = str(v) if v == v and v is not None else "<unknown>"
            groups.setdefault(g, []).append(i)
        agg = np.zeros((X_raw.shape[0], len(groups)), dtype=np.float32)
        for j, (gname, idxs) in enumerate(sorted(groups.items())):
            agg[:, j] = X_raw[:, idxs].sum(axis=1)
            group_keys.append((lvl, gname))
        blocks.append(agg)
    if not blocks:
        return np.zeros((X_raw.shape[0], 0), dtype=np.float32), group_keys
    return np.concatenate(blocks, axis=1), group_keys


def _species_feat(X_raw: np.ndarray, transform: str) -> np.ndarray:
    """Per-species compositional transform for the VAE input."""
    if transform == "clr":
        Xp = X_raw + 1e-6
        lg = np.log(Xp)
        return (lg - lg.mean(axis=1, keepdims=True)).astype(np.float32)
    return np.log1p(X_raw).astype(np.float32)


def build_vae_input(
    X_raw: np.ndarray, feat_clades: List[str], taxonomy, transform: str,
    levels: Tuple[str, ...],
) -> np.ndarray:
    """Construct the (unscaled) VAE input: [CLR/log1p species | log1p aggregates]."""
    agg, _ = build_taxon_aggregates(X_raw, feat_clades, taxonomy, levels)
    return np.concatenate(
        [_species_feat(X_raw, transform), np.log1p(agg)], axis=1
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CAPDAVAE(nn.Module):
    """Encoder/decoder MLP with a supervised class head on the latent mean."""

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


def _conditional_alignment(mu, y, dom, n_classes):
    """First-moment per-(study, class) latent spread around the class centroid."""
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
        study_means = [mu_c[dom_c == d].mean(0) for d in present
                       if (dom_c == d).sum() >= 1]
        if len(study_means) < 2:
            continue
        sm = torch.stack(study_means, 0)
        centroid = sm.mean(0, keepdim=True)
        total = total + ((sm - centroid) ** 2).sum(1).mean()
        n_terms += 1
    return total / n_terms if n_terms else mu.new_zeros(())


def _conditional_cov_alignment(mu, y, dom, n_classes):
    """Second-moment (covariance) conditional alignment — conditional CORAL."""
    total = mu.new_zeros(())
    n_terms = 0
    for c in range(n_classes):
        cmask = (y == c)
        if cmask.sum() < 4:
            continue
        mu_c = mu[cmask]
        dom_c = dom[cmask]
        covs = [torch.cov(mu_c[dom_c == d].T)
                for d in torch.unique(dom_c) if (dom_c == d).sum() >= 2]
        if len(covs) < 2:
            continue
        ref = torch.stack(covs, 0).mean(0)
        for cov in covs:
            total = total + ((cov - ref) ** 2).mean()
        n_terms += 1
    return total / n_terms if n_terms else mu.new_zeros(())


# --------------------------------------------------------------------------- #
# Single-VAE training
# --------------------------------------------------------------------------- #
def train_one_vae(
    Xin: np.ndarray, y: np.ndarray, dom: np.ndarray, n_classes: int,
    *, seed: int, device: str = "cpu", hp: Optional[Dict] = None,
) -> CAPDAVAE:
    """Train a single CAPDA-VAE on pre-built, *pre-scaled* input ``Xin``.

    ``y`` are contiguous class indices ``0..n_classes-1``; ``dom`` are integer
    study indices.  Returns the trained (eval-mode) model.
    """
    assert torch is not None, "torch is required to train CAPDA-VAE"
    p = {**DEFAULTS, **(hp or {})}
    rng = np.random.RandomState(seed)
    torch.manual_seed(int(seed))
    dev = torch.device(device)

    Xt = torch.tensor(Xin, dtype=torch.float32, device=dev)
    yt = torch.tensor(y.astype(np.int64), device=dev)
    dt = torch.tensor(dom.astype(np.int64), device=dev)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    cw = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1.0)),
                      device=dev, dtype=torch.float32)

    model = CAPDAVAE(Xin.shape[1], n_classes, p["latent"], p["hidden"],
                     p["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"],
                            weight_decay=p["weight_decay"])
    n = Xin.shape[0]
    bs = min(256, n)
    epochs = int(p["epochs"])
    model.train()
    for ep in range(epochs):
        w_kl = p["beta"] * min(1.0, ep / max(1, int(epochs * p["kl_warmup"])))
        w_al = p["gamma"] * min(1.0, ep / max(1, int(epochs * p["align_warmup"])))
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb, yb, db = Xt[idx], yt[idx], dt[idx]
            recon, mu, lv, logits, _z = model(xb)
            rec = F.mse_loss(recon, xb, reduction="none").sum(1).mean()
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
            ce = F.cross_entropy(logits, yb, weight=cw)
            al = _conditional_alignment(mu, yb, db, n_classes)
            loss = rec + w_kl * kl + p["alpha"] * ce + w_al * al
            if p["gamma_cov"] > 0:
                cov_al = _conditional_cov_alignment(mu, yb, db, n_classes)
                loss = loss + (w_al / max(p["gamma"], 1e-8)) * p["gamma_cov"] * cov_al
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def vae_class_probs(model: CAPDAVAE, Xin: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Softmax class-head probabilities on (pre-scaled) input ``Xin``."""
    dev = torch.device(device)
    with torch.no_grad():
        x = torch.tensor(Xin, dtype=torch.float32, device=dev)
        logits = model.cls(model.encode(x)[0])
        return torch.softmax(logits, 1).cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Fit (train + domain-aware OOF) and encode — produce stacking embeddings
# --------------------------------------------------------------------------- #
def _embedding_columns(n_species: int, n_classes: int) -> List[str]:
    return ([f"{FEAT_COL_PREFIX}{i}" for i in range(n_species)]
            + [f"{PROB_COL_PREFIX}{i}" for i in range(n_classes)])


def capda_fit(
    X_raw: np.ndarray, sample_ids: List[str], feat_clades: List[str],
    study: np.ndarray, y_raw: np.ndarray, taxonomy,
    *, seed: int = 42, device: str = "cpu", hp: Optional[Dict] = None,
) -> Tuple["pd.DataFrame", Dict, Dict]:  # noqa: F821
    """Train CAPDA-VAE with domain-aware OOF stacking on the given studies.

    Returns ``(embeddings_df, final_state_dict, config)`` where ``embeddings_df``
    is ``[log1p-species | OOF invariant-prob]`` for *all* input samples, the
    state dict is the final VAE (trained on all labelled samples) for the encode
    step, and ``config`` carries everything needed to rebuild + apply it.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    p = {**DEFAULTS, **(hp or {})}
    levels = tuple(p["agg_levels"])
    transform = str(p["transform"])

    study = np.asarray(study)
    # label encoding over labelled samples (missing -> excluded from fitting)
    y_str = np.array(["" if (v is None or (isinstance(v, float) and v != v))
                      else str(v) for v in y_raw], dtype=object)
    labeled = y_str != ""
    classes = sorted(set(y_str[labeled].tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    if n_classes < 2:
        raise SystemExit(
            "capda-vae: need >=2 disease classes among labelled training "
            f"samples, found {n_classes}."
        )
    y = np.array([cls_to_idx.get(v, -1) for v in y_str], dtype=np.int64)
    studies = sorted(set(study.tolist()))
    s2i = {s: i for i, s in enumerate(studies)}
    dom = np.array([s2i[s] for s in study], dtype=np.int64)

    Xin_full = build_vae_input(X_raw, feat_clades, taxonomy, transform, levels)
    n_samples = X_raw.shape[0]
    oof = np.zeros((n_samples, n_classes), dtype=np.float32)

    # domain-aware OOF: for each study, train on the labelled OTHER studies and
    # predict this study (incl. its unlabelled rows).
    for s in studies:
        in_te = (study == s)
        in_tr = (~in_te) & labeled
        if in_tr.sum() < 10 or in_te.sum() < 1:
            continue
        yt = y[in_tr]
        uniq = sorted(set(yt.tolist()))
        if len(uniq) < 2:
            continue
        remap = {c: i for i, c in enumerate(uniq)}
        yt_c = np.array([remap[c] for c in yt], dtype=np.int64)
        sc = StandardScaler().fit(Xin_full[in_tr])
        m = train_one_vae(
            sc.transform(Xin_full[in_tr]).astype(np.float32), yt_c, dom[in_tr],
            len(uniq), seed=seed, device=device, hp=p)
        probs = vae_class_probs(m, sc.transform(Xin_full[in_te]).astype(np.float32),
                                device)
        full = np.zeros((int(in_te.sum()), n_classes), dtype=np.float32)
        for j, c in enumerate(uniq):
            full[:, c] = probs[:, j]
        oof[in_te] = full

    # final VAE on all labelled samples — used to encode the held-out cohort.
    sc_final = StandardScaler().fit(Xin_full[labeled])
    final = train_one_vae(
        sc_final.transform(Xin_full[labeled]).astype(np.float32), y[labeled],
        dom[labeled], n_classes, seed=seed, device=device, hp=p)

    # Fallback for any rows the OOF loop skipped (tiny / unlabelled-only study):
    # use the final model's probabilities so no embedding row is left empty.
    empty = oof.sum(axis=1) <= 0.0
    if empty.any():
        oof[empty] = vae_class_probs(
            final, sc_final.transform(Xin_full[empty]).astype(np.float32), device)

    species = np.log1p(X_raw).astype(np.float32)
    emb = np.concatenate([species, oof], axis=1)
    columns = _embedding_columns(species.shape[1], n_classes)
    df = pd.DataFrame(emb, index=list(sample_ids), columns=columns)

    config = {
        "model_type": "capda-vae",
        "n_species": int(species.shape[1]),
        "input_dim": int(Xin_full.shape[1]),
        "n_classes": int(n_classes),
        "class_order": classes,
        "transform": transform,
        "agg_levels": list(levels),
        "latent": int(p["latent"]),
        "hidden": int(p["hidden"]),
        "dropout": float(p["dropout"]),
        "gamma_cov": float(p["gamma_cov"]),
        "vae_epochs": int(p["epochs"]),
        "scaler_mean": sc_final.mean_.astype(np.float64).tolist(),
        "scaler_scale": sc_final.scale_.astype(np.float64).tolist(),
        "prob_col_prefix": PROB_COL_PREFIX,
        "feat_col_prefix": FEAT_COL_PREFIX,
    }
    return df, final.state_dict(), config


# --------------------------------------------------------------------------- #
# Single-study fit (leak-free stratified-K-fold OOF) + reusable helpers
# --------------------------------------------------------------------------- #
def build_capda_from_config(config: Dict) -> CAPDAVAE:
    """Reconstruct an (untrained) :class:`CAPDAVAE` from a saved ``config``.

    Shared by ``biomevae-embed`` / ``biomevae-test`` / ``biomevae-interpret`` so
    every consumer rebuilds the network with identical dimensions.
    """
    assert torch is not None, "torch is required to build CAPDA-VAE"
    return CAPDAVAE(
        int(config["input_dim"]), int(config["n_classes"]),
        int(config.get("latent", config.get("latent_dim", 32))),
        int(config["hidden"]), float(config["dropout"]),
    )


def capda_scale(Xin: np.ndarray, config: Dict) -> np.ndarray:
    """Apply the StandardScaler stored in ``config`` to a built VAE input."""
    mean = np.asarray(config["scaler_mean"], dtype=np.float32)
    scale = np.asarray(config["scaler_scale"], dtype=np.float32)
    return ((Xin - mean) / scale).astype(np.float32)


class CAPDARawEncoderMean(nn.Module):
    """Map *raw per-species counts* to the latent mean ``mu``.

    ``biomevae-interpret`` runs SHAP's KernelExplainer, which perturbs the
    raw per-species count vector (``otu_names`` order).  This wrapper rebuilds
    the multi-resolution ``[CLR/log1p species | log1p aggregates]`` VAE input,
    applies the stored scaler, and returns the encoder mean — so the generic
    SHAP machinery treats CAPDA exactly like any other ``encode``-able VAE.
    """

    def __init__(self, model: CAPDAVAE, feat_clades: List[str], taxonomy,
                 config: Dict):
        super().__init__()
        self.model = model
        self.feat_clades = list(feat_clades)
        self.taxonomy = taxonomy
        self.transform = str(config.get("transform", DEFAULTS["transform"]))
        self.levels = tuple(config.get("agg_levels", DEFAULTS["agg_levels"]))
        self.register_buffer(
            "_mean", torch.tensor(np.asarray(config["scaler_mean"], np.float32)))
        self.register_buffer(
            "_scale", torch.tensor(np.asarray(config["scaler_scale"], np.float32)))

    def forward(self, x_counts: torch.Tensor) -> torch.Tensor:
        x_np = x_counts.detach().cpu().numpy().astype(np.float32)
        Xin = build_vae_input(
            x_np, self.feat_clades, self.taxonomy, self.transform, self.levels)
        Xt = torch.tensor(Xin, dtype=torch.float32, device=x_counts.device)
        Xs = (Xt - self._mean) / self._scale
        mu, _ = self.model.encode(Xs)
        return mu


def capda_fit_single_study(
    X_raw: np.ndarray, sample_ids: List[str], feat_clades: List[str],
    y_raw: np.ndarray, taxonomy, *, study: Optional[np.ndarray] = None,
    n_splits: int = 5, seed: int = 42, device: str = "cpu",
    hp: Optional[Dict] = None,
) -> Tuple["pd.DataFrame", Dict, Dict]:  # noqa: F821
    """Train CAPDA-VAE on a **single study** with leak-free OOF stacking.

    The cross-study conditional alignment of :func:`capda_fit` is meaningless
    when every sample comes from one cohort, so here it falls back gracefully:

    * the domain-aware OOF (train-on-other-studies) is replaced by a
      **stratified K-fold** OOF that is the within-study analogue — each sample
      is scored by a VAE that never saw it, so the class-probability columns
      handed to the downstream XGBoost are not in-sample-leaky;
    * the conditional first/second-moment alignment terms are still wired in
      and *re-activate automatically* if a within-study sub-cohort label is
      supplied via ``study`` (``>= 2`` levels); with a single level they are
      identically zero (see :func:`_conditional_alignment`).

    Returns ``(embeddings_df, final_state_dict, config)`` with the same
    ``[log1p-species | OOF-prob]`` layout as :func:`capda_fit`.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    p = {**DEFAULTS, **(hp or {})}
    levels = tuple(p["agg_levels"])
    transform = str(p["transform"])

    y_str = np.array(["" if (v is None or (isinstance(v, float) and v != v))
                      else str(v) for v in y_raw], dtype=object)
    labeled = y_str != ""
    classes = sorted(set(y_str[labeled].tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    if n_classes < 2:
        raise SystemExit(
            "capda-vae (single-study): need >=2 disease classes among "
            f"labelled samples, found {n_classes}."
        )
    y = np.array([cls_to_idx.get(v, -1) for v in y_str], dtype=np.int64)

    # Optional within-study sub-cohort domain. A single level (the common case
    # for one study) leaves the conditional alignment terms identically zero.
    if study is None:
        dom = np.zeros(len(sample_ids), dtype=np.int64)
        studies: List[str] = ["<single>"]
    else:
        study = np.asarray(study)
        studies = sorted(set(study.tolist()))
        s2i = {s: i for i, s in enumerate(studies)}
        dom = np.array([s2i[s] for s in study], dtype=np.int64)

    Xin_full = build_vae_input(X_raw, feat_clades, taxonomy, transform, levels)
    n_samples = X_raw.shape[0]
    oof = np.zeros((n_samples, n_classes), dtype=np.float32)

    # Leak-free out-of-fold probabilities via stratified K-fold over the
    # labelled samples (the within-study analogue of LOSO's per-study holdout).
    lab_idx = np.where(labeled)[0]
    y_lab = y[lab_idx]
    min_class = int(np.min(np.bincount(y_lab, minlength=n_classes)))
    k_eff = max(2, min(int(n_splits), min_class)) if min_class >= 2 else 0
    folds = []
    if k_eff >= 2:
        skf = StratifiedKFold(n_splits=k_eff, shuffle=True,
                              random_state=int(seed))
        folds = list(skf.split(lab_idx, y_lab))

    for tr_local, te_local in folds:
        tr = lab_idx[tr_local]
        te = lab_idx[te_local]
        uniq = sorted(set(y[tr].tolist()))
        if len(uniq) < 2:
            continue
        remap = {c: i for i, c in enumerate(uniq)}
        yt_c = np.array([remap[c] for c in y[tr]], dtype=np.int64)
        sc = StandardScaler().fit(Xin_full[tr])
        m = train_one_vae(
            sc.transform(Xin_full[tr]).astype(np.float32), yt_c, dom[tr],
            len(uniq), seed=seed, device=device, hp=p)
        probs = vae_class_probs(
            m, sc.transform(Xin_full[te]).astype(np.float32), device)
        full = np.zeros((len(te), n_classes), dtype=np.float32)
        for j, c in enumerate(uniq):
            full[:, c] = probs[:, j]
        oof[te] = full

    # Final VAE on all labelled samples — used by the embed/test/encode steps.
    sc_final = StandardScaler().fit(Xin_full[labeled])
    final = train_one_vae(
        sc_final.transform(Xin_full[labeled]).astype(np.float32), y[labeled],
        dom[labeled], n_classes, seed=seed, device=device, hp=p)

    # Fallback for rows the OOF loop could not score (unlabelled, or too few
    # per-class samples for CV): use the final model's in-sample probabilities.
    empty = oof.sum(axis=1) <= 0.0
    if empty.any():
        oof[empty] = vae_class_probs(
            final, sc_final.transform(Xin_full[empty]).astype(np.float32), device)

    species = np.log1p(X_raw).astype(np.float32)
    emb = np.concatenate([species, oof], axis=1)
    columns = _embedding_columns(species.shape[1], n_classes)
    df = pd.DataFrame(emb, index=list(sample_ids), columns=columns)

    config = {
        "model_type": "capda-vae",
        "single_study": True,
        "n_species": int(species.shape[1]),
        "input_dim": int(Xin_full.shape[1]),
        "n_classes": int(n_classes),
        "class_order": classes,
        "transform": transform,
        "agg_levels": list(levels),
        "latent": int(p["latent"]),
        # generic CLIs (biomevae-test capacity loss) read ``latent_dim``.
        "latent_dim": int(p["latent"]),
        "hidden": int(p["hidden"]),
        "dropout": float(p["dropout"]),
        "gamma_cov": float(p["gamma_cov"]),
        "vae_epochs": int(p["epochs"]),
        "n_splits_effective": int(k_eff),
        "n_domains": int(len(studies)),
        "feature_clades": list(feat_clades),
        "scaler_mean": sc_final.mean_.astype(np.float64).tolist(),
        "scaler_scale": sc_final.scale_.astype(np.float64).tolist(),
        "prob_col_prefix": PROB_COL_PREFIX,
        "feat_col_prefix": FEAT_COL_PREFIX,
    }
    return df, final.state_dict(), config


def capda_encode(
    X_raw: np.ndarray, sample_ids: List[str], feat_clades: List[str],
    taxonomy, state_dict: Dict, config: Dict, *, device: str = "cpu",
) -> "pd.DataFrame":  # noqa: F821
    """Apply the final CAPDA-VAE to new (held-out) samples.

    Produces the same ``[log1p-species | final-VAE-prob]`` columns the trainer
    wrote, so the concatenated train+holdout table feeds the standard
    classifier unchanged.
    """
    import pandas as pd

    levels = tuple(config.get("agg_levels", DEFAULTS["agg_levels"]))
    transform = str(config.get("transform", DEFAULTS["transform"]))
    Xin = build_vae_input(X_raw, feat_clades, taxonomy, transform, levels)
    input_dim = int(config["input_dim"])
    if Xin.shape[1] != input_dim:
        raise SystemExit(
            "capda-vae encode: VAE input dim mismatch "
            f"(holdout {Xin.shape[1]} vs trained {input_dim}). The held-out "
            "fold must share the merged clade set + taxonomy with training."
        )
    mean = np.asarray(config["scaler_mean"], dtype=np.float32)
    scale = np.asarray(config["scaler_scale"], dtype=np.float32)
    Xin_s = ((Xin - mean) / scale).astype(np.float32)

    n_classes = int(config["n_classes"])
    model = CAPDAVAE(input_dim, n_classes, int(config["latent"]),
                     int(config["hidden"]), float(config["dropout"]))
    model.load_state_dict(state_dict)
    model.eval().to(torch.device(device))
    probs = vae_class_probs(model, Xin_s, device)

    n_species = int(config["n_species"])
    species = np.log1p(X_raw).astype(np.float32)
    if species.shape[1] != n_species:
        raise SystemExit(
            "capda-vae encode: species feature count mismatch "
            f"(holdout {species.shape[1]} vs trained {n_species})."
        )
    emb = np.concatenate([species, probs], axis=1)
    columns = _embedding_columns(n_species, n_classes)
    return pd.DataFrame(emb, index=list(sample_ids), columns=columns)
