"""Regression tests for ``biomevae.cli.loso_classify.evaluate_loso_fold``.

The held-out study can carry classes that are absent from the training
fold (or vice versa).  XGBoost requires contiguous ``[0, K-1]`` labels,
so the evaluator must remap the training labels and translate
predictions / probabilities back into the global class space before
computing metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from biomevae.cli.loso_classify import evaluate_loso_fold


def _toy_dataset(rng: np.random.Generator, *, n_per_class: int = 20, dim: int = 6):
    """Four well-separated Gaussian classes across three studies."""
    centers = np.array([
        [+3.0, +3.0, 0, 0, 0, 0],
        [-3.0, +3.0, 0, 0, 0, 0],
        [-3.0, -3.0, 0, 0, 0, 0],
        [+3.0, -3.0, 0, 0, 0, 0],
    ])[:, :dim]
    Xs, ys, studies = [], [], []
    # Study A: classes 0, 1
    for c in (0, 1):
        Xs.append(centers[c] + rng.normal(scale=0.4, size=(n_per_class, dim)))
        ys.extend([c] * n_per_class)
        studies.extend(["A"] * n_per_class)
    # Study B: classes 0, 1, 3 (covers class 3 so it exists in training when
    # study C is held out).
    for c in (0, 1, 3):
        Xs.append(centers[c] + rng.normal(scale=0.4, size=(n_per_class, dim)))
        ys.extend([c] * n_per_class)
        studies.extend(["B"] * n_per_class)
    # Study C: only class 2 — when held out, training fold lacks class 2.
    Xs.append(centers[2] + rng.normal(scale=0.4, size=(n_per_class, dim)))
    ys.extend([2] * n_per_class)
    studies.extend(["C"] * n_per_class)
    X = np.vstack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    study = np.asarray(studies)
    return X, y, study


def test_evaluate_loso_fold_handles_noncontiguous_train_classes():
    """Training fold missing a global class should not crash XGBoost.

    Holding out study C means y_tr ∈ {0, 1, 3} (the global LabelEncoder
    space is {0, 1, 2, 3}).  The pre-fix code passed those labels straight
    to XGBoost and crashed with::

        ValueError: Invalid classes inferred from unique values of `y`.
                    Expected: [0 1 2 3], got [0 1 3]
    """
    rng = np.random.default_rng(0)
    X, y, study = _toy_dataset(rng)
    class_names = ["healthy", "CRC", "adenoma", "IBD"]

    # Sanity: training fold is missing class 2, eval fold is class 2 only.
    train = study != "C"
    assert set(np.unique(y[train]).tolist()) == {0, 1, 3}
    assert set(np.unique(y[~train]).tolist()) == {2}

    results = evaluate_loso_fold(
        X=X, y=y, study=study,
        held_out="C", class_names=class_names, seeds=[0],
    )
    summary = results["XGBoost"]
    # Confusion matrix is (n_classes, n_classes) regardless of which classes
    # were present in either fold.
    assert np.asarray(summary["confusion_matrix"]).shape == (4, 4)
    # The model can never predict class 2 (it never saw it), so balanced
    # accuracy on a held-out fold of only class 2 should be 0.
    assert summary["balanced_accuracy"] == pytest.approx(0.0)
    assert summary["f1_macro"] >= 0.0
    assert summary["n_features"] == X.shape[1]


def test_evaluate_loso_fold_predictions_use_global_label_space():
    """Predicted class IDs must come from the *global* label set even
    when the training fold drops some of those classes — i.e. the
    XGBoost-internal contiguous indices must be back-mapped."""
    rng = np.random.default_rng(1)
    # Training fold ends up with classes {0, 2} (non-contiguous).  The
    # back-map must turn XGBoost's 0/1 predictions into global 0/2.
    n = 25
    centers = np.array([[+3.0, 0.0], [-3.0, 0.0], [0.0, +3.0]])
    X = np.vstack([
        centers[0] + rng.normal(scale=0.3, size=(n, 2)),  # class 0, study A
        centers[2] + rng.normal(scale=0.3, size=(n, 2)),  # class 2, study A
        centers[1] + rng.normal(scale=0.3, size=(n, 2)),  # class 1, study B (held out)
    ]).astype(np.float32)
    y = np.array([0] * n + [2] * n + [1] * n, dtype=np.int64)
    study = np.array(["A"] * (2 * n) + ["B"] * n)

    results = evaluate_loso_fold(
        X=X, y=y, study=study,
        held_out="B", class_names=["c0", "c1", "c2"], seeds=[0],
    )
    summary = results["XGBoost"]
    # Confusion matrix has the full (3, 3) shape; rows for true classes
    # 0 and 2 are empty since the eval fold only contains class 1.
    cm = np.asarray(summary["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert cm[0].sum() == 0 and cm[2].sum() == 0
    assert cm[1].sum() == n
    # Predictions must lie in the global label set {0, 1, 2}; class 1
    # is impossible (training never saw it), so populated columns must
    # come from {0, 2}.  Without the back-map XGBoost would have
    # returned values from {0, 1} that mean "global 0 or global 2".
    populated_cols = np.flatnonzero(cm[1] > 0).tolist()
    assert set(populated_cols).issubset({0, 2})


def test_evaluate_loso_fold_rejects_single_class_training_fold():
    """If the training fold collapses to a single class, raise instead of
    handing XGBoost an undefined problem."""
    rng = np.random.default_rng(2)
    n = 20
    X = rng.normal(size=(2 * n, 4)).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    study = np.array(["A"] * n + ["B"] * n)
    with pytest.raises(ValueError, match="only 1 class"):
        evaluate_loso_fold(
            X=X, y=y, study=study,
            held_out="B", class_names=["healthy", "CRC"], seeds=[0],
        )
