"""Unit tests for :func:`biomevae.loso.coral_align`.

Per-study CORAL alignment is the preprocessing step that powers the
``xgb-coral`` LOSO row.  These tests verify the two contracts the LOSO
pipeline depends on:

1. After alignment, the Frobenius distance between per-study
   covariance matrices collapses to ~0 — that's the entire point.
2. Each study's transformation depends only on its own samples; mixing
   in or removing samples from *another* study leaves the first study's
   aligned values unchanged.  This is what guarantees no class-label
   leakage from the held-out study at LOSO time.
"""
from __future__ import annotations

import numpy as np
import pytest

from biomevae.loso import coral_align, covariance_frobenius


def _two_studies(seed: int = 0):
    """Build two synthetic studies with deliberately different (mu, Sigma)."""
    rng = np.random.RandomState(seed)
    p = 6
    # Study A: anisotropic covariance, non-zero mean.
    A_cov = np.diag([4.0, 1.0, 0.25, 4.0, 1.0, 0.25])
    A_mu = np.array([2.0, -1.0, 0.5, 0.0, 1.5, -0.5])
    XA = rng.multivariate_normal(A_mu, A_cov, size=200)
    # Study B: rotated covariance, different mean.
    R = np.linalg.qr(rng.standard_normal((p, p)))[0]
    B_cov = R @ np.diag([0.5, 2.0, 0.5, 2.0, 0.5, 2.0]) @ R.T
    B_mu = np.array([-1.0, 2.0, -2.0, 1.0, -0.5, 0.5])
    XB = rng.multivariate_normal(B_mu, B_cov, size=200)
    X = np.vstack([XA, XB])
    study = np.array(["A"] * 200 + ["B"] * 200, dtype=object)
    return X, study


def test_coral_align_collapses_covariance_distance():
    X, study = _two_studies(seed=7)
    XA = X[study == "A"]
    XB = X[study == "B"]
    pre = covariance_frobenius(XA, XB)

    X_aligned, _ = coral_align(X, study, ridge=1e-3, reference="mean")
    XA_a = X_aligned[study == "A"]
    XB_a = X_aligned[study == "B"]
    post = covariance_frobenius(XA_a, XB_a)

    # Two-orders-of-magnitude shrinkage is the contract; in practice the
    # remaining drift comes only from the ridge regularisation.
    assert post < pre / 100.0, (
        f"CORAL alignment failed to collapse covariance distance: "
        f"pre={pre:.4f}, post={post:.4f}."
    )

    # Means should also align (CORAL here includes a centring step).
    assert np.allclose(XA_a.mean(axis=0), XB_a.mean(axis=0), atol=1e-3)


def test_coral_align_per_study_transform_is_local():
    """Adding a third study must not perturb the first two studies' aligned values."""
    X, study = _two_studies(seed=3)
    rng = np.random.RandomState(11)
    XC = rng.multivariate_normal(
        mean=np.zeros(6), cov=np.eye(6) * 3.0, size=120,
    )
    X_three = np.vstack([X, XC])
    study_three = np.concatenate([study, np.array(["C"] * 120, dtype=object)])

    aligned_two, _ = coral_align(X, study, reference="identity")
    aligned_three, _ = coral_align(X_three, study_three, reference="identity")
    # With reference="identity", every study is whitened to N(0, I) using
    # only its own statistics, so adding study C does not change A or B.
    assert np.allclose(aligned_two, aligned_three[: X.shape[0]], atol=1e-6)


def test_coral_align_underdetermined_study_uses_ridge():
    """n_samples < n_features should not crash thanks to the ridge regulariser."""
    rng = np.random.RandomState(0)
    p = 50
    n_per_study = 8  # < p, so np.cov is rank-deficient
    X = rng.standard_normal((n_per_study * 3, p))
    study = np.array(
        ["A"] * n_per_study + ["B"] * n_per_study + ["C"] * n_per_study,
        dtype=object,
    )
    X_aligned, stats = coral_align(X, study, ridge=1e-2, reference="mean")
    assert X_aligned.shape == X.shape
    assert np.isfinite(X_aligned).all()
    # Per-study book-keeping should still report sample counts.
    assert stats["A"]["n"] == n_per_study
    assert "_reference" in stats


def test_coral_align_reference_largest_picks_most_samples():
    rng = np.random.RandomState(1)
    p = 4
    X_small = rng.standard_normal((20, p))
    X_big = rng.standard_normal((100, p)) * 5.0 + 3.0
    X = np.vstack([X_small, X_big])
    study = np.array(["small"] * 20 + ["big"] * 100, dtype=object)
    _, stats = coral_align(X, study, reference="largest")
    # The largest-cohort branch copies that study's own (mu, Sigma) into
    # the reference, so the reference statistics should match study "big".
    np.testing.assert_allclose(
        stats["_reference"]["mu"], stats["big"]["mu"], atol=1e-9,
    )
    np.testing.assert_allclose(
        stats["_reference"]["cov"], stats["big"]["cov"], atol=1e-9,
    )


def test_coral_align_rejects_mismatched_study_length():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="len\\(study\\)"):
        coral_align(X, ["A", "A", "B"], ridge=1e-3)


def test_coral_align_rejects_unknown_reference():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="reference"):
        coral_align(X, ["A", "A", "B", "B"], reference="oops")
