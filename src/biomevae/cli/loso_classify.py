"""Leave-one-study-out classifier evaluation on biomevae embeddings.

Trains an XGBoost classifier on training-fold latent embeddings and
evaluates it on the held-out study.  The embeddings file must be the
``embeddings.tsv`` produced by an unsupervised or DIVA-style training
run on the *merged* multi-study dataset.

For DIVA models you can either evaluate the full latent
(``embeddings.tsv``, default), the class-anchored slice
(``embeddings_z_y.tsv``), or the residual slice
(``embeddings_z_x.tsv``).  Pass ``--latent-slice z_y`` / ``z_x`` to do
so.

Output: a JSON file with the same schema as
``classification_results.json`` so the LOSO aggregation rule can plug
into the existing aggregator without code changes.

Usage::

    biomevae-loso-classify \\
        --embeddings out/tree-dtm-vae/embeddings.tsv \\
        --metadata   merged_sample_metadata.tsv \\
        --held-out-study FengQ_2015 \\
        --label disease --outdir out/classify_loso/tree-dtm-vae/FengQ_2015
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from biomevae.utils import set_global_seed


def _resolve_embeddings_path(
    embeddings: str, slice_name: Optional[str],
) -> str:
    """Return the path corresponding to ``--latent-slice`` (full / z_y / z_x)."""
    if not slice_name or slice_name == "full":
        return embeddings
    base = Path(embeddings)
    sliced = base.with_name(f"embeddings_{slice_name}.tsv")
    if sliced.exists():
        return str(sliced)
    raise FileNotFoundError(
        f"loso-classify: requested slice '{slice_name}' not found at {sliced}.  "
        "Re-run training with the DIVA CLI to populate the per-factor "
        "embedding files."
    )


def _load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.values.astype(np.float32), list(df.index)


def _load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "sample_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sample_id"})
    return df.set_index("sample_id")


def evaluate_loso_fold(
    *,
    X: np.ndarray,
    y: np.ndarray,
    study: np.ndarray,
    held_out: str,
    class_names: List[str],
    seeds: Sequence[int],
) -> Dict[str, dict]:
    """XGBoost on (study != held_out), tested on (study == held_out)."""
    eval_mask = study == held_out
    train_mask = ~eval_mask
    if not eval_mask.any():
        raise ValueError(
            f"loso-classify: held-out study '{held_out}' contributes 0 samples."
        )
    if not train_mask.any():
        raise ValueError(
            f"loso-classify: leaving out '{held_out}' empties the training set."
        )

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_ev, y_ev = X[eval_mask], y[eval_mask]
    n_classes = len(class_names)
    all_labels = list(range(n_classes))

    # The global LabelEncoder produces class IDs in [0, n_classes).  After
    # holding out a study, the training fold may be missing some IDs (e.g.
    # global classes [0,1,2,3,4] but y_tr ∈ {0,1,3,4}).  XGBoost requires
    # contiguous [0, K-1] labels, so remap y_tr → [0, K_train) for fitting
    # and map predictions/probabilities back to the global class space.
    train_classes = np.unique(y_tr).astype(int)
    n_train_classes = int(train_classes.size)
    if n_train_classes < 2:
        raise ValueError(
            f"loso-classify: training fold for held-out '{held_out}' has "
            f"only {n_train_classes} class(es); need at least 2 to train."
        )
    if n_train_classes < n_classes:
        missing = [class_names[c] for c in all_labels if c not in train_classes]
        print(
            f"[loso-classify] training fold is missing class(es) "
            f"{missing}; remapping {n_train_classes}/{n_classes} classes "
            f"present in training to contiguous IDs for XGBoost."
        )
    train_to_xgb = {int(c): i for i, c in enumerate(train_classes)}
    xgb_to_train = {i: int(c) for c, i in train_to_xgb.items()}
    y_tr_xgb = np.asarray(
        [train_to_xgb[int(v)] for v in y_tr], dtype=np.int64,
    )

    # Same preprocessing as biomevae.classify (StandardScaler) so the LOSO
    # results are directly comparable to the within-study CV table.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_ev_s = scaler.transform(X_ev)

    per_seed = []
    for seed in seeds:
        set_global_seed(int(seed))
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=int(seed), n_jobs=1,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr_xgb)
        clf.fit(X_tr_s, y_tr_xgb, sample_weight=sample_weight)
        y_pred_xgb = clf.predict(X_ev_s)
        y_pred = np.asarray(
            [xgb_to_train[int(v)] for v in y_pred_xgb], dtype=np.int64,
        )

        acc = float(accuracy_score(y_ev, y_pred))
        bacc = float(balanced_accuracy_score(y_ev, y_pred))
        f1 = float(f1_score(
            y_ev, y_pred, labels=all_labels,
            average="macro", zero_division=0,
        ))
        auroc: Optional[float] = None
        if hasattr(clf, "predict_proba"):
            proba_xgb = clf.predict_proba(X_ev_s)
            # Expand back to the global class space; columns for classes
            # absent from training stay zero.
            proba = np.zeros(
                (proba_xgb.shape[0], n_classes), dtype=proba_xgb.dtype,
            )
            for xgb_idx, orig_idx in xgb_to_train.items():
                proba[:, orig_idx] = proba_xgb[:, xgb_idx]
            try:
                if n_classes == 2:
                    auroc = float(roc_auc_score(y_ev, proba[:, 1]))
                elif proba.shape[1] == n_classes:
                    auroc = float(roc_auc_score(
                        y_ev, proba, labels=all_labels,
                        multi_class="ovr", average="macro",
                    ))
            except (ValueError, IndexError):
                pass

        per_seed.append({
            "seed": int(seed),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1,
            "auroc": auroc,
            "y_pred": y_pred.tolist(),
        })

    bacc_arr = np.array([r["balanced_accuracy"] for r in per_seed])
    f1_arr = np.array([r["f1_macro"] for r in per_seed])
    auroc_arr = np.array([
        r["auroc"] for r in per_seed if r["auroc"] is not None
    ])

    cm = confusion_matrix(
        y_ev, np.array(per_seed[0]["y_pred"]), labels=all_labels,
    )
    report = classification_report(
        y_ev, np.array(per_seed[0]["y_pred"]),
        labels=all_labels, target_names=class_names, zero_division=0,
    )

    summary = {
        "classifier_name": "XGBoost",
        "held_out_study": held_out,
        "n_train_samples": int(train_mask.sum()),
        "n_eval_samples": int(eval_mask.sum()),
        "n_features": int(X.shape[1]),
        "class_names": class_names,
        "seeds": [int(r["seed"]) for r in per_seed],
        "balanced_accuracy": float(bacc_arr.mean()),
        "f1_macro": float(f1_arr.mean()),
        "auroc": float(auroc_arr.mean()) if auroc_arr.size > 0 else None,
        "across_seed_std": {
            "balanced_accuracy": float(bacc_arr.std(ddof=1)) if bacc_arr.size > 1 else None,
            "f1_macro": float(f1_arr.std(ddof=1)) if f1_arr.size > 1 else None,
            "auroc": float(auroc_arr.std(ddof=1)) if auroc_arr.size > 1 else None,
        },
        "per_seed_metrics": {
            str(r["seed"]): {
                "balanced_accuracy": r["balanced_accuracy"],
                "f1_macro": r["f1_macro"],
                "auroc": r["auroc"],
            }
            for r in per_seed
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return {"XGBoost": summary}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-loso-classify")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--label", default="disease")
    ap.add_argument("--study-col", default="study_name")
    ap.add_argument("--held-out-study", required=True)
    ap.add_argument(
        "--latent-slice", default="full",
        choices=["full", "z_y", "z_x", "z_d"],
        help=(
            "Which latent factor to evaluate.  'full' uses embeddings.tsv "
            "(works for any model); 'z_y' / 'z_x' / 'z_d' require a DIVA "
            "training run that wrote the per-factor TSVs."
        ),
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    emb_path = _resolve_embeddings_path(args.embeddings, args.latent_slice)
    print(f"[loso-classify] embeddings: {emb_path}")
    print(f"[loso-classify] held-out:  {args.held_out_study}")
    print(f"[loso-classify] seeds:     {args.seeds}")

    X, sample_ids = _load_embeddings(emb_path)
    metadata = _load_metadata(args.metadata)
    by_id = metadata.reindex(sample_ids)
    if args.study_col not in by_id.columns:
        raise SystemExit(
            f"metadata lacks '{args.study_col}' column."
        )
    if args.label not in by_id.columns:
        raise SystemExit(
            f"metadata lacks label column '{args.label}'."
        )

    label_str = by_id[args.label].astype(str).fillna("").values
    valid = label_str != ""
    if not valid.all():
        n_drop = int((~valid).sum())
        print(f"[loso-classify] dropping {n_drop} samples with missing labels.")
    X = X[valid]
    sample_ids_v = [s for s, v in zip(sample_ids, valid) if v]
    label_str = label_str[valid]
    study_str = by_id[args.study_col].astype(str).values[valid]

    le = LabelEncoder()
    y = le.fit_transform(label_str)
    class_names = list(le.classes_)

    results = evaluate_loso_fold(
        X=X, y=y, study=study_str,
        held_out=args.held_out_study,
        class_names=class_names,
        seeds=list(args.seeds),
    )

    out_path = outdir / "classification_results.json"
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    summary = results["XGBoost"]
    print(f"[loso-classify] balanced_accuracy={summary['balanced_accuracy']:.4f}")
    print(f"[loso-classify] f1_macro={summary['f1_macro']:.4f}")
    if summary["auroc"] is not None:
        print(f"[loso-classify] auroc={summary['auroc']:.4f}")
    print(f"[loso-classify] wrote {out_path}")


if __name__ == "__main__":
    main()
