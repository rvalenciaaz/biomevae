"""Fast standalone strict-LOSO harness for the CRC cohort benchmark.

Reproduces the strict leave-one-study-out (LOSO) evaluation used by the
Snakemake pipeline, but as a single self-contained script that runs in a
couple of minutes on CPU.  It exists so we can iterate quickly on a NEW model
that aims to resolve the tension between *taxonomy* (phylogenetic inductive
bias) and *domain invariance* (cross-study generalisation), comparing it
head-to-head against the published baselines under identical preprocessing and
evaluation.

Data : per_study/<study>/{sgb_table.tsv, sample_metadata.tsv, phyla.tsv}
Task : multi-class disease prediction (control / CRC / adenoma /
       carcinoma_surgery_history) trained on N-1 studies, evaluated on the
       held-out study.
Metric: balanced_accuracy and f1_macro (mean +/- std over eval seeds), exactly
        as in src/biomevae/classify.py.

Models are pluggable via the MODELS registry at the bottom; each is a callable
``fit_predict(Xtr, ytr, dtr, Xte, seed) -> y_pred`` where ``dtr`` is the
per-sample study (domain) index of the training rows.  The baseline mirrors
classify.py (log1p -> StandardScaler -> XGBoost).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
PER_STUDY = REPO / "per_study"

CRC_STUDIES = [
    "FengQ_2015", "VogtmannE_2016", "ZellerG_2014", "YuJ_2015", "WirbelJ_2018",
    "YachidaS_2019", "ThomasAM_2018a", "ThomasAM_2018b", "ThomasAM_2019_c",
    "HanniganGD_2017", "GuptaA_2019",
]
EVAL_SEEDS = [int(s) for s in os.environ.get("HARNESS_SEEDS", "0,1,2").split(",")]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_all() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return (X_relab, y_disease, study, taxonomy).

    X_relab : (n_samples, n_features) relative abundances, union of all SGB
              clades across studies, missing filled with 0.
    y        : disease label per sample.
    study    : study name per sample.
    taxonomy : phyla.tsv lineage table indexed by clade_name (union).
    """
    cols: List[pd.Series] = []
    ys: List[str] = []
    studies: List[str] = []
    sample_ids: List[str] = []
    tax_frames = []
    for s in CRC_STUDIES:
        sgb = pd.read_csv(PER_STUDY / s / "sgb_table.tsv", sep="\t", index_col=0)
        if "NCBI_tax_id" in sgb.columns:
            sgb = sgb.drop(columns=["NCBI_tax_id"])
        sgb = sgb.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        meta = pd.read_csv(PER_STUDY / s / "sample_metadata.tsv", sep="\t")
        meta = meta.set_index("sample_id")
        common = [c for c in sgb.columns if c in meta.index]
        sgb = sgb[common]
        for sid in common:
            disease = meta.loc[sid, "disease"]
            # Drop samples without a valid disease label (6 rows in
            # VogtmannE_2016); a missing label is not a prediction target and
            # would otherwise appear as a spurious extra class.
            if disease is None or (isinstance(disease, float) and disease != disease) \
                    or str(disease).lower() in ("nan", "none", ""):
                continue
            cols.append(sgb[sid].rename(f"{s}::{sid}"))
            ys.append(str(disease))
            studies.append(s)
            sample_ids.append(f"{s}::{sid}")
        tax = pd.read_csv(PER_STUDY / s / "phyla.tsv", sep="\t", index_col=0)
        tax_frames.append(tax)
    X = pd.concat(cols, axis=1).T.fillna(0.0)        # samples x features
    X.index = sample_ids
    X = X.sort_index(axis=1)
    y = pd.Series(ys, index=sample_ids, name="disease")
    study = pd.Series(studies, index=sample_ids, name="study")
    taxonomy = pd.concat(tax_frames).groupby(level=0).first()
    return X, y, study, taxonomy


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model_fn: Callable, X: pd.DataFrame, y: pd.Series, study: pd.Series,
             taxonomy: pd.DataFrame, model_name: str,
             verbose: bool = True) -> pd.DataFrame:
    le = LabelEncoder().fit(y.values)
    feat = X.columns
    rows = []
    for held in CRC_STUDIES:
        te_mask = (study == held).values
        tr_mask = ~te_mask
        Xtr = X.values[tr_mask]
        Xte = X.values[te_mask]
        ytr_g = le.transform(y.values[tr_mask])
        yte = le.transform(y.values[te_mask])
        dtr = study.values[tr_mask]
        # Remap training labels to a contiguous 0..K-1 space (a held-out study
        # may carry classes absent from training, and vice versa -> classifiers
        # like XGBoost require contiguous labels). Predictions are mapped back
        # to the global label space for scoring; classes never seen in training
        # are simply never predicted (correctly scoring recall 0 on that fold).
        fold_le = LabelEncoder().fit(ytr_g)
        ytr = fold_le.transform(ytr_g)
        baccs, f1s = [], []
        for seed in EVAL_SEEDS:
            y_pred_local = model_fn(Xtr, ytr, dtr, Xte, seed,
                                    feat=feat, taxonomy=taxonomy)
            y_pred = fold_le.inverse_transform(np.asarray(y_pred_local))
            baccs.append(balanced_accuracy_score(yte, y_pred))
            f1s.append(f1_score(yte, y_pred, average="macro"))
        rows.append({
            "model": model_name, "held_out_study": held,
            "balanced_accuracy": float(np.mean(baccs)),
            "balanced_accuracy_std": float(np.std(baccs)),
            "f1_macro": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)),
            "n_train_samples": int(tr_mask.sum()),
            "n_eval_samples": int(te_mask.sum()),
        })
        if verbose:
            r = rows[-1]
            print(f"  {held:18s} bacc={r['balanced_accuracy']:.4f} "
                  f"f1={r['f1_macro']:.4f}")
    df = pd.DataFrame(rows)
    if verbose:
        print(f"  {model_name} MEAN bacc={df['balanced_accuracy'].mean():.4f} "
              f"f1={df['f1_macro'].mean():.4f}")
    return df


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Bounded thread pool: on a 32-core box, n_jobs=-1 oversubscribes badly on these
# small (~1500 x ~935) problems and competes with torch. Cap it.
N_JOBS = int(os.environ.get("HARNESS_NJOBS", "8"))


def _xgb(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=int(seed), n_jobs=N_JOBS,
        eval_metric="mlogloss",
    )


def model_xgb_baseline(Xtr, ytr, dtr, Xte, seed, **kw):
    """Mirror of src/biomevae/classify.py: log1p -> StandardScaler -> XGBoost."""
    Xtr = np.log1p(Xtr)
    Xte = np.log1p(Xte)
    sc = StandardScaler().fit(Xtr)
    clf = _xgb(seed).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def model_capda_vae(Xtr, ytr, dtr, Xte, seed, **kw):
    """CAPDA-VAE: taxonomy multi-resolution features + class-conditional latent
    alignment, latent (+head probs) classified by XGBoost. See capda_vae.py."""
    from capda_vae import capda_vae_fit_predict
    return capda_vae_fit_predict(Xtr, ytr, dtr, Xte, seed,
                                 feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_head(Xtr, ytr, dtr, Xte, seed, **kw):
    """Same VAE, but predict directly from the supervised class head."""
    from capda_vae import capda_vae_fit_predict
    return capda_vae_fit_predict(Xtr, ytr, dtr, Xte, seed, classifier="head",
                                 feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_hybrid(Xtr, ytr, dtr, Xte, seed, **kw):
    """Augment-not-replace: XGBoost on [raw multi-res features + latent + head
    probs]. The conditionally-aligned latent adds invariance to the full
    discriminative set."""
    from capda_vae import capda_vae_fit_predict
    return capda_vae_fit_predict(Xtr, ytr, dtr, Xte, seed, classifier="hybrid",
                                 feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_rawlatent(Xtr, ytr, dtr, Xte, seed, **kw):
    """Leakage-free augmentation: XGBoost on [baseline log1p-species + aligned
    latent], no in-sample head probs, no aggregates. Isolates the latent's
    marginal value over the raw-feature baseline."""
    from capda_vae import capda_vae_fit_predict
    return capda_vae_fit_predict(Xtr, ytr, dtr, Xte, seed, classifier="raw_latent",
                                 feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_oof(Xtr, ytr, dtr, Xte, seed, **kw):
    """Domain-aware OOF stacking: XGBoost on [raw log1p-species + out-of-fold
    VAE head-probs + latent]. Leak-free; see capda_vae_oof_fit_predict."""
    from capda_vae import capda_vae_oof_fit_predict
    return capda_vae_oof_fit_predict(Xtr, ytr, dtr, Xte, seed,
                                     feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_oof_probs(Xtr, ytr, dtr, Xte, seed, **kw):
    """OOF stacking with the distilled invariant prediction ONLY: XGBoost on
    [raw log1p-species + OOF head-probs], no latent. Minimal noise footprint to
    avoid perturbing the booster on raw-friendly folds."""
    from capda_vae import capda_vae_oof_fit_predict
    return capda_vae_oof_fit_predict(Xtr, ytr, dtr, Xte, seed, use_latent=False,
                                     feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_oof_probs_cov(Xtr, ytr, dtr, Xte, seed, **kw):
    """probs-only OOF stack + second-moment (covariance) conditional alignment
    (conditional CORAL) to strengthen the invariant prediction."""
    from capda_vae import capda_vae_oof_fit_predict
    return capda_vae_oof_fit_predict(Xtr, ytr, dtr, Xte, seed, use_latent=False,
                                     gamma_cov=2.0,
                                     feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_oof_probs_cov4(Xtr, ytr, dtr, Xte, seed, **kw):
    """Champion (probs-only OOF + covariance alignment) with stronger covariance
    weight (gamma_cov=4) to probe whether more conditional CORAL widens the win."""
    from capda_vae import capda_vae_oof_fit_predict
    return capda_vae_oof_fit_predict(Xtr, ytr, dtr, Xte, seed, use_latent=False,
                                     gamma_cov=4.0,
                                     feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_oof_probs_cov_ens(Xtr, ytr, dtr, Xte, seed, **kw):
    """Champion (probs-only OOF + covariance alignment) with the invariant
    prediction seed-ensembled over 4 VAE seeds to cut its variance before
    stacking. Run with HARNESS_SEEDS=0 (the internal ensemble provides
    stability) to keep cost ~1 sweep."""
    from capda_vae import capda_vae_oof_fit_predict
    return capda_vae_oof_fit_predict(Xtr, ytr, dtr, Xte, seed, use_latent=False,
                                     gamma_cov=2.0, n_vae_seeds=4,
                                     feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_oof_probs_cov_clr(Xtr, ytr, dtr, Xte, seed, **kw):
    """Champion (probs-only OOF + covariance alignment) with the VAE input in
    CLR (centered-log-ratio) compositional coordinates instead of log1p."""
    from capda_vae import capda_vae_oof_fit_predict
    return capda_vae_oof_fit_predict(Xtr, ytr, dtr, Xte, seed, use_latent=False,
                                     gamma_cov=2.0, transform="clr",
                                     feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_oof_probs_cov_clr_ens(Xtr, ytr, dtr, Xte, seed, **kw):
    """Champion (CLR + covariance OOF stack) with the invariant prediction
    seed-ensembled over 4 VAE seeds. Combines the two winning levers (CLR +
    variance reduction). Run with HARNESS_SEEDS=0."""
    from capda_vae import capda_vae_oof_fit_predict
    return capda_vae_oof_fit_predict(Xtr, ytr, dtr, Xte, seed, use_latent=False,
                                     gamma_cov=2.0, transform="clr", n_vae_seeds=4,
                                     feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_multiview(Xtr, ytr, dtr, Xte, seed, **kw):
    """Multi-view OOF stacking: raw + OOF invariant probs from a species-CLR
    view AND a coarse-taxonomy view. Run with HARNESS_SEEDS=0."""
    from capda_vae import capda_vae_multiview_fit_predict
    return capda_vae_multiview_fit_predict(Xtr, ytr, dtr, Xte, seed,
                                           gamma_cov=2.0, transform="clr",
                                           views=("full", "coarse"),
                                           feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


def model_capda_vae_ensemble(Xtr, ytr, dtr, Xte, seed, **kw):
    """Inner-LOSO-weighted probability blend of raw-XGB + aligned VAE.
    See capda_vae_ensemble_fit_predict."""
    from capda_vae import capda_vae_ensemble_fit_predict
    return capda_vae_ensemble_fit_predict(Xtr, ytr, dtr, Xte, seed,
                                          feat=kw.get("feat"), taxonomy=kw.get("taxonomy"))


MODELS: Dict[str, Callable] = {
    "xgb-baseline": model_xgb_baseline,
    "capda-vae": model_capda_vae,
    "capda-vae-head": model_capda_vae_head,
    "capda-vae-hybrid": model_capda_vae_hybrid,
    "capda-vae-rawlatent": model_capda_vae_rawlatent,
    "capda-vae-oof": model_capda_vae_oof,
    "capda-vae-oof-probs": model_capda_vae_oof_probs,
    "capda-vae-oof-probs-cov": model_capda_vae_oof_probs_cov,
    "capda-vae-oof-probs-cov4": model_capda_vae_oof_probs_cov4,
    "capda-vae-oof-probs-cov-ens": model_capda_vae_oof_probs_cov_ens,
    "capda-vae-oof-probs-cov-clr": model_capda_vae_oof_probs_cov_clr,
    "capda-vae-oof-probs-cov-clr-ens": model_capda_vae_oof_probs_cov_clr_ens,
    "capda-vae-multiview": model_capda_vae_multiview,
    "capda-vae-ensemble": model_capda_vae_ensemble,
}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS),
                    help="model keys to run")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    print("Loading CRC cohorts ...")
    X, y, study, taxonomy = load_all()
    print(f"  X={X.shape}  classes={sorted(y.unique())}  studies={study.nunique()}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    all_df = []
    for name in args.models:
        if name not in MODELS:
            print(f"!! unknown model {name}; have {list(MODELS)}")
            continue
        print(f"\n== {name} ==")
        df = evaluate(MODELS[name], X, y, study, taxonomy, name)
        df.to_csv(outdir / f"loso_{name}.tsv", sep="\t", index=False)
        all_df.append(df)
    if all_df:
        merged = pd.concat(all_df, ignore_index=True)
        merged.to_csv(outdir / "loso_all.tsv", sep="\t", index=False)
        summary = (merged.groupby("model")[["balanced_accuracy", "f1_macro"]]
                   .agg(["mean", "median", "std"]))
        print("\n==== SUMMARY ====")
        print(summary.to_string())
        summary.to_csv(outdir / "summary.tsv", sep="\t")


if __name__ == "__main__":
    main()
