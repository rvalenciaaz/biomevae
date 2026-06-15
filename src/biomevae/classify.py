"""Classification of sample metadata using learned VAE embeddings.

Provides utilities to train and evaluate an XGBoost classifier on latent
embeddings produced by any biomevae model, with proper cross-validation and
reporting for publication.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .utils import capture_provenance, set_global_seed

#: Canonical list of 5 evaluation seeds used across the repository.  Every
#: model evaluation (classification, reconstruction benchmarking, NMF baseline,
#: enterosignature scoring, …) is repeated across these seeds for
#: reproducibility.  Override via the ``--seeds`` CLI flag or the ``seeds``
#: keyword argument of the corresponding ``evaluate_*`` function.
DEFAULT_EVAL_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)


def normalise_seeds(
    seeds: Optional[Iterable[int]] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """Resolve a seed specification into a concrete list of integer seeds.

    Precedence rules:
        1. If ``seeds`` is provided (non-empty iterable), use it as-is.
        2. Else if a scalar ``seed`` is provided, return ``[seed]``.
        3. Else return the canonical :data:`DEFAULT_EVAL_SEEDS`.

    Raises ``ValueError`` if ``seeds`` is an empty iterable.
    """
    if seeds is not None:
        resolved = [int(s) for s in seeds]
        if not resolved:
            raise ValueError(
                "seeds must contain at least one integer; "
                "pass None to use DEFAULT_EVAL_SEEDS."
            )
        return resolved
    if seed is not None:
        return [int(seed)]
    return list(DEFAULT_EVAL_SEEDS)


@dataclass
class ClassificationResult:
    """Results from a classifier evaluation, pooled across one or more seeds.

    When ``seeds`` contains more than one entry the scalar fields
    (``accuracy``, ``balanced_accuracy``, …) are the *mean of the per-seed
    means* — i.e. each seed contributes a single value equal to the
    average of its ``n_splits * n_repeats`` folds, and the top-level
    number is the mean of those five per-seed means.  ``across_seed_std``
    contains the unbiased standard deviation of the same per-seed means
    and is the correct measure of the model's run-to-run variance in the
    sense of Bouthillier et al. (2021).

    ``per_fold_*`` lists still concatenate every seed's fold metrics
    (``n_splits * n_repeats * n_seeds`` entries) for use in paired
    significance tests and violin plots.  ``per_seed_metrics`` keeps the
    per-seed summaries verbatim.  ``metadata['provenance']`` embeds the
    git SHA, package versions and platform info captured by
    :func:`biomevae.utils.capture_provenance`.
    """

    classifier_name: str
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    auroc: Optional[float]  # None if >2 classes and predict_proba unavailable
    per_fold_accuracy: List[float]
    per_fold_balanced_accuracy: List[float]
    per_fold_f1_macro: List[float]
    confusion_matrix: np.ndarray
    classification_report: str
    class_names: List[str]
    n_samples: int
    n_features: int
    seeds: List[int] = field(default_factory=list)
    per_seed_metrics: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    across_seed_std: Dict[str, Optional[float]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "classifier_name": self.classifier_name,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "auroc": self.auroc,
            "per_fold_accuracy": self.per_fold_accuracy,
            "per_fold_balanced_accuracy": self.per_fold_balanced_accuracy,
            "per_fold_f1_macro": self.per_fold_f1_macro,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "class_names": self.class_names,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "seeds": list(self.seeds),
            "per_seed_metrics": {
                str(k): dict(v) for k, v in self.per_seed_metrics.items()
            },
            "across_seed_std": dict(self.across_seed_std),
            "metadata": self.metadata,
        }


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings TSV produced by biomevae-embed.

    Returns (X [n_samples, latent_dim], sample_names).
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.values.astype(np.float32), list(df.index)


def load_metadata(path: str, label_col: str = "disease") -> Tuple[pd.Series, List[str]]:
    """Load metadata TSV and extract a label column.

    Returns (labels Series indexed by sample_id, sample_ids list).
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in metadata. "
            f"Available: {list(df.columns)}"
        )
    return df[label_col], list(df.index)


def align_embeddings_metadata(
    embeddings: np.ndarray,
    emb_samples: List[str],
    labels: pd.Series,
    meta_samples: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """Align embeddings and metadata by sample_id, drop NaN labels.

    Returns (X, y_encoded, class_names, label_encoder).
    """
    emb_df = pd.DataFrame(embeddings, index=emb_samples)
    common = emb_df.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("No common sample IDs between embeddings and metadata.")

    emb_df = emb_df.loc[common]
    lab = labels.loc[common].dropna()
    # Drop samples with missing labels
    emb_df = emb_df.loc[lab.index]

    le = LabelEncoder()
    y = le.fit_transform(lab.values)
    return emb_df.values.astype(np.float32), y, list(le.classes_), le


def _get_classifiers(seed: int, *, n_jobs: int = 1) -> Dict[str, object]:
    """Return a dict of classifier name -> fitted sklearn estimator.

    Parameters
    ----------
    seed:
        Random state passed to every estimator that accepts one.  This is
        critical for reproducibility: without it XGBoost re-rolls the same
        ``subsample=0.8`` / ``colsample_bytree=0.8`` masks on every call and
        every seed, which means the "5-seed" protocol only varies the CV
        splits, not the classifier itself.
    n_jobs:
        Number of parallel workers.  Defaults to ``1`` because XGBoost's
        histogram reduction with ``n_jobs=-1`` depends on thread scheduling
        and introduces non-determinism across runs even when ``random_state``
        is pinned.  Callers willing to trade reproducibility for speed can
        override this, but downstream evaluations in this repo do not.
    """
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=int(seed), n_jobs=int(n_jobs),
        ),
    }


def _evaluate_classifiers_single_seed(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    classifier_names: Optional[List[str]],
) -> Dict[str, ClassificationResult]:
    """Run one repeated-stratified-k-fold evaluation pass at a single seed."""
    # Seed every RNG (Python, NumPy, PyTorch, cuDNN, PYTHONHASHSEED,
    # CUBLAS_WORKSPACE_CONFIG) before any sklearn/xgboost operation.  This
    # nails down not just the CV splits but also the XGBoost bootstrap masks,
    # the StandardScaler internals and any downstream operation that reads
    # from the NumPy legacy global state.
    set_global_seed(int(seed))

    all_classifiers = _get_classifiers(int(seed), n_jobs=1)
    if classifier_names:
        classifiers = {k: v for k, v in all_classifiers.items() if k in classifier_names}
    else:
        classifiers = all_classifiers

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=int(seed)
    )

    results: Dict[str, ClassificationResult] = {}

    for clf_name, clf_template in classifiers.items():
        fold_acc, fold_bacc, fold_f1 = [], [], []
        all_y_true, all_y_pred, all_y_proba = [], [], []

        for train_idx, test_idx in rskf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            from sklearn.base import clone
            clf = clone(clf_template)
            # Balance the training loss across classes so the classifier's
            # objective matches the reported metric (balanced_accuracy). Without
            # this, XGBoost (and any cost-sensitive classifier we may add) biases
            # toward majority classes and the minority-class recall is
            # systematically under-reported — the effect visible in the
            # MetaCardis_2020_a confusion matrices, where HF/IGT/CAD are almost
            # never predicted. ``balanced_accuracy_score`` on the test split
            # remains unweighted (line below), so no double-counting occurs.
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            clf.fit(X_train_s, y_train, sample_weight=sample_weight)
            y_pred = clf.predict(X_test_s)

            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_bacc.append(balanced_accuracy_score(y_test, y_pred))
            fold_f1.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

            if hasattr(clf, "predict_proba"):
                all_y_proba.extend(clf.predict_proba(X_test_s).tolist())

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        # Compute aggregate metrics
        acc = float(np.mean(fold_acc))
        bacc = float(np.mean(fold_bacc))
        f1_mac = float(np.mean(fold_f1))
        f1_w = float(f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0))

        # AUROC
        auroc: Optional[float] = None
        if all_y_proba:
            y_proba = np.array(all_y_proba)
            n_classes = len(class_names)
            try:
                if n_classes == 2:
                    auroc = float(roc_auc_score(all_y_true, y_proba[:, 1]))
                else:
                    auroc = float(roc_auc_score(
                        all_y_true, y_proba, multi_class="ovr", average="macro"
                    ))
            except (ValueError, IndexError):
                pass

        cm = confusion_matrix(all_y_true, all_y_pred)
        report = classification_report(
            all_y_true, all_y_pred, target_names=class_names, zero_division=0
        )

        results[clf_name] = ClassificationResult(
            classifier_name=clf_name,
            accuracy=acc,
            balanced_accuracy=bacc,
            f1_macro=f1_mac,
            f1_weighted=f1_w,
            auroc=auroc,
            per_fold_accuracy=fold_acc,
            per_fold_balanced_accuracy=fold_bacc,
            per_fold_f1_macro=fold_f1,
            confusion_matrix=cm,
            classification_report=report,
            class_names=class_names,
            n_samples=len(X),
            n_features=X.shape[1],
            seeds=[int(seed)],
            per_seed_metrics={
                str(int(seed)): {
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                    "f1_macro": f1_mac,
                    "f1_weighted": f1_w,
                    "auroc": auroc,
                }
            },
        )

    return results


def _across_seed_std(values: Sequence[Optional[float]]) -> Optional[float]:
    """Unbiased std across the per-seed means; ``None`` for <2 valid seeds."""
    cleaned = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(cleaned) < 2:
        return None
    return float(np.std(np.asarray(cleaned, dtype=float), ddof=1))


def _aggregate_classifier_results(
    per_seed: Sequence[Dict[str, ClassificationResult]],
    seeds: Sequence[int],
) -> Dict[str, ClassificationResult]:
    """Merge per-seed classifier results into one with across-seed stats.

    The top-level scalar metrics are the mean of the per-seed means
    (one sample per seed), and ``across_seed_std`` captures the unbiased
    standard deviation of those per-seed means.  This is the Bouthillier
    et al. (2021) "account for run-to-run variance" framing and gives
    numbers that are directly comparable across methods even when each
    seed runs a different number of folds.

    ``per_fold_*`` lists still concatenate every seed's fold metrics so
    that downstream paired tests / violin plots can look at fold-level
    variation.
    """
    if not per_seed:
        raise ValueError("per_seed must contain at least one entry")
    first = per_seed[0]
    merged: Dict[str, ClassificationResult] = {}
    for clf_name in first.keys():
        entries = [ps[clf_name] for ps in per_seed]
        pooled_fold_acc: List[float] = []
        pooled_fold_bacc: List[float] = []
        pooled_fold_f1: List[float] = []
        per_seed_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        seed_acc: List[float] = []
        seed_bacc: List[float] = []
        seed_f1_macro: List[float] = []
        seed_f1_weighted: List[float] = []
        seed_auroc: List[Optional[float]] = []
        cm_sum: Optional[np.ndarray] = None
        for seed, entry in zip(seeds, entries):
            pooled_fold_acc.extend(entry.per_fold_accuracy)
            pooled_fold_bacc.extend(entry.per_fold_balanced_accuracy)
            pooled_fold_f1.extend(entry.per_fold_f1_macro)
            per_seed_metrics[str(int(seed))] = {
                "accuracy": entry.accuracy,
                "balanced_accuracy": entry.balanced_accuracy,
                "f1_macro": entry.f1_macro,
                "f1_weighted": entry.f1_weighted,
                "auroc": entry.auroc,
            }
            seed_acc.append(float(entry.accuracy))
            seed_bacc.append(float(entry.balanced_accuracy))
            seed_f1_macro.append(float(entry.f1_macro))
            seed_f1_weighted.append(float(entry.f1_weighted))
            seed_auroc.append(entry.auroc)
            cm = np.asarray(entry.confusion_matrix)
            cm_sum = cm.copy() if cm_sum is None else cm_sum + cm

        def _mean_or_zero(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        mean_acc = _mean_or_zero(seed_acc)
        mean_bacc = _mean_or_zero(seed_bacc)
        mean_f1_macro = _mean_or_zero(seed_f1_macro)
        mean_f1_weighted = _mean_or_zero(seed_f1_weighted)
        auroc_valid = [v for v in seed_auroc if v is not None]
        mean_auroc: Optional[float] = (
            float(np.mean(auroc_valid)) if auroc_valid else None
        )

        across_seed_std = {
            "accuracy": _across_seed_std(seed_acc),
            "balanced_accuracy": _across_seed_std(seed_bacc),
            "f1_macro": _across_seed_std(seed_f1_macro),
            "f1_weighted": _across_seed_std(seed_f1_weighted),
            "auroc": _across_seed_std(seed_auroc),
        }

        report_header = (
            f"Classification report from seed {int(seeds[0])} "
            f"(aggregated over {len(seeds)} seeds: "
            f"{', '.join(str(int(s)) for s in seeds)}).\n"
        )
        merged[clf_name] = ClassificationResult(
            classifier_name=clf_name,
            accuracy=mean_acc,
            balanced_accuracy=mean_bacc,
            f1_macro=mean_f1_macro,
            f1_weighted=mean_f1_weighted,
            auroc=mean_auroc,
            per_fold_accuracy=pooled_fold_acc,
            per_fold_balanced_accuracy=pooled_fold_bacc,
            per_fold_f1_macro=pooled_fold_f1,
            confusion_matrix=cm_sum if cm_sum is not None else np.zeros((0, 0), dtype=int),
            classification_report=report_header + entries[0].classification_report,
            class_names=entries[0].class_names,
            n_samples=entries[0].n_samples,
            n_features=entries[0].n_features,
            seeds=[int(s) for s in seeds],
            per_seed_metrics=per_seed_metrics,
            across_seed_std=across_seed_std,
            metadata={
                "seeds": [int(s) for s in seeds],
                "n_seeds": len(seeds),
                "n_splits": None,
                "n_repeats": None,
                "aggregation": "mean_of_per_seed_means",
            },
        )
    return merged


def evaluate_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: Optional[int] = None,
    seeds: Optional[Iterable[int]] = None,
    classifier_names: Optional[List[str]] = None,
) -> Dict[str, ClassificationResult]:
    """Evaluate multiple classifiers using repeated stratified k-fold CV.

    The evaluation is repeated once per entry in ``seeds`` (defaulting to
    :data:`DEFAULT_EVAL_SEEDS`, i.e. 5 seeds) and the per-seed results are
    pooled together into a single :class:`ClassificationResult` per
    classifier.  Per-seed scalar metrics are preserved under
    ``per_seed_metrics`` for reproducibility reporting.

    Parameters
    ----------
    X : array [n_samples, n_features]
        Feature matrix (e.g. VAE embeddings).
    y : array [n_samples]
        Integer-encoded labels.
    class_names : list of str
        Human-readable class names matching label encoding.
    n_splits : int
        Number of CV folds per seed.
    n_repeats : int
        Number of CV repetitions per seed.
    seed : int, optional
        Legacy alias: if provided and ``seeds`` is not, a single seed is used.
        New code should prefer ``seeds``.
    seeds : iterable of int, optional
        Seeds to repeat the evaluation over.  Defaults to
        :data:`DEFAULT_EVAL_SEEDS` (5 seeds).
    classifier_names : list of str, optional
        Subset of classifiers to evaluate. If None, all are used.

    Returns
    -------
    dict mapping classifier name to ClassificationResult pooled over seeds.
    """
    resolved_seeds = normalise_seeds(seeds, seed)
    per_seed = [
        _evaluate_classifiers_single_seed(
            X, y, class_names, n_splits, n_repeats, s, classifier_names,
        )
        for s in resolved_seeds
    ]
    merged = _aggregate_classifier_results(per_seed, resolved_seeds)
    # Capture provenance once (same for every classifier in this call).
    provenance = capture_provenance(seeds=resolved_seeds)
    for res in merged.values():
        res.metadata["n_splits"] = n_splits
        res.metadata["n_repeats"] = n_repeats
        res.metadata["provenance"] = provenance
    return merged


def evaluate_embedding_classification(
    embeddings_path: str,
    metadata_path: str,
    label_col: str = "disease",
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: Optional[int] = None,
    seeds: Optional[Iterable[int]] = None,
    classifier_names: Optional[List[str]] = None,
) -> Dict[str, ClassificationResult]:
    """End-to-end: load embeddings + metadata, evaluate classifiers.

    ``seeds`` defaults to :data:`DEFAULT_EVAL_SEEDS` (5 seeds).

    Parameters
    ----------
    embeddings_path : str
        Path to embeddings.tsv from biomevae-embed.
    metadata_path : str
        Path to sample_metadata.tsv.
    label_col : str
        Metadata column to classify (default: "disease").
    """
    X_emb, emb_samples = load_embeddings(embeddings_path)
    labels, meta_samples = load_metadata(metadata_path, label_col)
    X, y, class_names, le = align_embeddings_metadata(
        X_emb, emb_samples, labels, meta_samples
    )
    return evaluate_classifiers(
        X, y, class_names,
        n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, seeds=seeds,
        classifier_names=classifier_names,
    )


def evaluate_direct_classification(
    sgb_table_path: str,
    metadata_path: str,
    label_col: str = "disease",
    log1p: bool = True,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: Optional[int] = None,
    seeds: Optional[Iterable[int]] = None,
    classifier_names: Optional[List[str]] = None,
) -> Dict[str, ClassificationResult]:
    """Baseline: XGBoost directly on the SGB abundance table (no VAE).

    ``seeds`` defaults to :data:`DEFAULT_EVAL_SEEDS` (5 seeds).

    Parameters
    ----------
    sgb_table_path : str
        Path to sgb_table.tsv (features x samples, with clade_name and
        NCBI_tax_id as the first two columns).
    metadata_path : str
        Path to sample_metadata.tsv.
    label_col : str
        Metadata column to classify (default: "disease").
    log1p : bool
        Apply log1p transform to abundances (default: True).
    """
    from biomevae.data import load_matrix

    X_raw, sample_names = load_matrix(sgb_table_path, log1p=log1p)
    labels, meta_samples = load_metadata(metadata_path, label_col)
    X, y, class_names, le = align_embeddings_metadata(
        X_raw, sample_names, labels, meta_samples
    )
    return evaluate_classifiers(
        X, y, class_names,
        n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, seeds=seeds,
        classifier_names=classifier_names,
    )


def save_classification_results(
    results: Dict[str, ClassificationResult],
    outdir: str,
    prefix: str = "",
) -> str:
    """Save classification results to JSON.

    Returns path to the saved JSON file.
    """
    os.makedirs(outdir, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""
    path = os.path.join(outdir, f"{pfx}classification_results.json")
    payload = {name: r.to_dict() for name, r in results.items()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def print_classification_summary(results: Dict[str, ClassificationResult]) -> None:
    """Print a formatted summary table of classification results."""
    print("=" * 80)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Classifier':<22} {'Accuracy':>10} {'Bal.Acc':>10} "
        f"{'F1-macro':>10} {'AUROC':>10}"
    )
    print("-" * 80)
    for name, r in results.items():
        auroc_str = f"{r.auroc:.4f}" if r.auroc is not None else "N/A"
        print(
            f"{name:<22} {r.accuracy:>10.4f} {r.balanced_accuracy:>10.4f} "
            f"{r.f1_macro:>10.4f} {auroc_str:>10}"
        )
        if r.seeds:
            seed_strs = ", ".join(str(s) for s in r.seeds)
            print(f"{'':<22}   pooled over {len(r.seeds)} seeds: [{seed_strs}]")
            bacc_std = r.across_seed_std.get("balanced_accuracy") if r.across_seed_std else None
            if len(r.seeds) > 1 and bacc_std is not None:
                print(
                    f"{'':<22}   balanced-acc across seeds: "
                    f"mean={r.balanced_accuracy:.4f} std={bacc_std:.4f}"
                )
    print("=" * 80)
