"""Tests for the 5-seed evaluation protocol.

Every post-training model evaluation in biomevae is repeated across the
seeds listed in :data:`biomevae.classify.DEFAULT_EVAL_SEEDS` (5 seeds) and
the per-seed summaries are aggregated with *across-seed statistics*
(mean of per-seed means, unbiased std of per-seed means) for
reproducibility.  These tests cover the core building blocks that make
that contract hold:

* ``biomevae.classify.evaluate_classifiers`` pools ``n_splits * n_repeats``
  fold metrics across every seed, records a ``per_seed_metrics`` dict,
  exposes ``across_seed_std`` and embeds a provenance block in
  ``metadata``.
* ``biomevae.reconstruction.cross_validate_nmf_multi_seed`` pools
  ``n_splits`` fold metrics across every seed and records per-seed
  summaries, pooled fold metrics, and a provenance block in
  ``metadata``.
* ``biomevae.reconstruction.merge_cross_val_results`` is the low-level
  helper used by the multi-seed wrappers and reports across-seed
  mean/std rather than pooled-fold mean/std.
* ``biomevae.reconstruction.compute_pairwise_seed_stats`` produces the
  paired Wilcoxon + Nadeau-Bengio corrected t-test used by the
  ``biomevae-pairwise-table`` CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomevae.classify import (  # noqa: E402
    DEFAULT_EVAL_SEEDS,
    evaluate_classifiers,
    normalise_seeds,
)
from biomevae.reconstruction import (  # noqa: E402
    CrossValResult,
    compute_pairwise_seed_stats,
    cross_validate_nmf_multi_seed,
    merge_cross_val_results,
)
from biomevae.utils import capture_provenance, set_global_seed  # noqa: E402


# --------------------------------------------------------------------------
# biomevae.classify
# --------------------------------------------------------------------------


def test_default_eval_seeds_has_five_unique_seeds() -> None:
    assert len(DEFAULT_EVAL_SEEDS) == 5
    assert len(set(DEFAULT_EVAL_SEEDS)) == 5


def test_normalise_seeds_defaults_to_five_seeds() -> None:
    assert normalise_seeds(None, None) == list(DEFAULT_EVAL_SEEDS)


def test_normalise_seeds_legacy_scalar_wins() -> None:
    assert normalise_seeds(None, 7) == [7]


def test_normalise_seeds_prefers_explicit_seeds_list() -> None:
    assert normalise_seeds([1, 2, 3], None) == [1, 2, 3]


def _synthetic_classification_data(n_samples: int = 60, n_features: int = 8):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    # Two linearly separable classes.
    y = np.zeros(n_samples, dtype=int)
    y[n_samples // 2 :] = 1
    X[n_samples // 2 :] += 1.5
    class_names = ["ctrl", "case"]
    return X, y, class_names


def test_evaluate_classifiers_pools_metrics_across_five_seeds() -> None:
    X, y, class_names = _synthetic_classification_data()
    n_splits = 3
    n_repeats = 2
    results = evaluate_classifiers(
        X, y, class_names,
        n_splits=n_splits,
        n_repeats=n_repeats,
        # Use the canonical 5-seed protocol.
    )

    assert results, "evaluate_classifiers returned no classifiers"
    for clf_name, res in results.items():
        # The pooled seed list must match DEFAULT_EVAL_SEEDS.
        assert res.seeds == list(DEFAULT_EVAL_SEEDS), (
            f"{clf_name}: unexpected seeds {res.seeds}"
        )
        # Per-seed metrics are preserved for reproducibility reporting.
        assert set(res.per_seed_metrics.keys()) == {
            str(int(s)) for s in DEFAULT_EVAL_SEEDS
        }
        # Fold metrics are the concatenation of per-seed folds:
        # (n_splits * n_repeats) folds per seed * 5 seeds.
        expected_folds = n_splits * n_repeats * len(DEFAULT_EVAL_SEEDS)
        assert len(res.per_fold_accuracy) == expected_folds
        assert len(res.per_fold_balanced_accuracy) == expected_folds
        assert len(res.per_fold_f1_macro) == expected_folds
        # Top-level scalar metrics are the *mean of per-seed means*, not
        # the pooled fold mean.  Verify that explicitly so a regression
        # back to naive pooling is caught here.
        seed_acc = [
            float(m["accuracy"]) for m in res.per_seed_metrics.values()
        ]
        seed_bacc = [
            float(m["balanced_accuracy"]) for m in res.per_seed_metrics.values()
        ]
        seed_f1 = [
            float(m["f1_macro"]) for m in res.per_seed_metrics.values()
        ]
        assert res.accuracy == pytest.approx(float(np.mean(seed_acc)), rel=1e-6)
        assert res.balanced_accuracy == pytest.approx(
            float(np.mean(seed_bacc)), rel=1e-6,
        )
        assert res.f1_macro == pytest.approx(
            float(np.mean(seed_f1)), rel=1e-6,
        )
        # ``across_seed_std`` captures the unbiased std of the per-seed
        # means (= the Bouthillier et al. 2021 run-to-run variance).
        expected_std = float(np.std(np.asarray(seed_bacc), ddof=1))
        assert res.across_seed_std["balanced_accuracy"] == pytest.approx(
            expected_std, rel=1e-6, abs=1e-9,
        )
        # Metadata records the seed list, size and aggregation mode.
        assert res.metadata["seeds"] == list(DEFAULT_EVAL_SEEDS)
        assert res.metadata["n_seeds"] == 5
        assert res.metadata["n_splits"] == n_splits
        assert res.metadata["n_repeats"] == n_repeats
        assert res.metadata["aggregation"] == "mean_of_per_seed_means"
        # A provenance block is embedded so that figures are traceable.
        provenance = res.metadata["provenance"]
        assert "captured_at" in provenance
        assert "packages" in provenance
        assert "platform" in provenance
        assert provenance["seeds"] == list(DEFAULT_EVAL_SEEDS)


def test_evaluate_classifiers_legacy_single_seed_alias_uses_one_seed() -> None:
    X, y, class_names = _synthetic_classification_data()
    results = evaluate_classifiers(
        X, y, class_names,
        n_splits=3,
        n_repeats=1,
        seed=7,
    )
    for res in results.values():
        assert res.seeds == [7]
        assert set(res.per_seed_metrics.keys()) == {"7"}


# --------------------------------------------------------------------------
# biomevae.reconstruction
# --------------------------------------------------------------------------


def test_merge_cross_val_results_pools_fold_metrics() -> None:
    a = CrossValResult(
        fold_metrics=[{"rmse": 1.0, "mae": 0.5}, {"rmse": 1.2, "mae": 0.6}],
        mean_metrics={"rmse": 1.1, "mae": 0.55},
        std_metrics={"rmse": 0.1, "mae": 0.05},
        metadata={"selected_rank": 4},
    )
    b = CrossValResult(
        fold_metrics=[{"rmse": 1.4, "mae": 0.7}, {"rmse": 1.6, "mae": 0.8}],
        mean_metrics={"rmse": 1.5, "mae": 0.75},
        std_metrics={"rmse": 0.1, "mae": 0.05},
        metadata={"selected_rank": 4},
    )
    merged = merge_cross_val_results([a, b], [42, 43])
    # Fold metrics still concatenate every seed's folds for paired tests.
    assert len(merged.fold_metrics) == 4
    # mean_metrics is now the *mean of per-seed means*, which for this
    # particular example equals the mean of pooled folds — but
    # std_metrics is the unbiased std over the per-seed means (0.2 here)
    # rather than the pooled-fold std (~0.25).
    assert merged.mean_metrics["rmse"] == pytest.approx(1.3, rel=1e-6)
    assert merged.mean_metrics["mae"] == pytest.approx(0.65, rel=1e-6)
    expected_rmse_std = float(np.std([1.1, 1.5], ddof=1))
    expected_mae_std = float(np.std([0.55, 0.75], ddof=1))
    assert merged.std_metrics["rmse"] == pytest.approx(expected_rmse_std, rel=1e-6)
    assert merged.std_metrics["mae"] == pytest.approx(expected_mae_std, rel=1e-6)
    assert merged.metadata["seeds"] == [42, 43]
    assert merged.metadata["n_seeds"] == 2
    assert merged.metadata["aggregation"] == "mean_of_per_seed_means"
    assert set(merged.metadata["per_seed_mean_metrics"].keys()) == {"42", "43"}
    assert merged.metadata["per_seed_mean_metrics"]["42"]["rmse"] == pytest.approx(
        1.1, rel=1e-6,
    )
    # ``pooled_fold_mean_metrics`` retains the old naive pooled mean.
    assert "pooled_fold_mean_metrics" in merged.metadata
    assert merged.metadata["pooled_fold_mean_metrics"]["rmse"] == pytest.approx(
        1.3, rel=1e-6,
    )
    # Provenance is captured.
    provenance = merged.metadata.get("provenance")
    assert provenance is not None
    assert provenance["seeds"] == [42, 43]


def test_compute_pairwise_seed_stats_uses_per_seed_means() -> None:
    # Two methods evaluated on five seeds each, with a consistent
    # advantage for method ``a`` (lower rmse).
    per_seed_a = {
        "per_seed_mean_metrics": {
            "42": {"rmse": 1.00},
            "43": {"rmse": 1.05},
            "44": {"rmse": 0.98},
            "45": {"rmse": 1.02},
            "46": {"rmse": 0.97},
        }
    }
    per_seed_b = {
        "per_seed_mean_metrics": {
            "42": {"rmse": 1.20},
            "43": {"rmse": 1.22},
            "44": {"rmse": 1.18},
            "45": {"rmse": 1.25},
            "46": {"rmse": 1.19},
        }
    }
    a = CrossValResult(
        fold_metrics=[],
        mean_metrics={"rmse": 1.004},
        std_metrics={"rmse": 0.03},
        metadata=per_seed_a,
    )
    b = CrossValResult(
        fold_metrics=[],
        mean_metrics={"rmse": 1.208},
        std_metrics={"rmse": 0.02},
        metadata=per_seed_b,
    )
    comps = compute_pairwise_seed_stats({"a": a, "b": b}, "rmse")
    assert len(comps) == 1
    row = comps[0]
    assert row["n"] == 5
    assert row["mean_diff"] < 0  # a is lower than b
    assert 0.0 <= row["p_value_sign"] <= 1.0
    assert 0.0 <= row["p_value_wilcoxon"] <= 1.0
    assert 0.0 <= row["p_value_tcorrected"] <= 1.0
    # Canonical p_value is the Nadeau-Bengio corrected t value.
    assert row["p_value"] == row["p_value_tcorrected"]


def test_cross_validate_nmf_multi_seed_pools_five_seeds() -> None:
    rng = np.random.default_rng(0)
    X = rng.poisson(5, size=(30, 40)).astype(float)
    n_splits = 3
    result = cross_validate_nmf_multi_seed(
        X,
        n_components=3,
        n_splits=n_splits,
        train_fraction=0.9,
        log1p=True,
        # Use the canonical 5-seed protocol.
    )
    # Pooled fold_metrics has n_splits * 5 entries.
    assert len(result.fold_metrics) == n_splits * len(DEFAULT_EVAL_SEEDS)
    # ``mean_metrics`` is the mean of the per-seed means — verify that
    # explicitly to pin the aggregation semantics.
    per_seed_means = result.metadata["per_seed_mean_metrics"]
    seed_rmse = [
        float(metrics["rmse"])
        for metrics in per_seed_means.values()
        if "rmse" in metrics
    ]
    if seed_rmse:
        assert result.mean_metrics["rmse"] == pytest.approx(
            float(np.mean(seed_rmse)), rel=1e-6,
        )
        expected_std = (
            float(np.std(np.asarray(seed_rmse), ddof=1))
            if len(seed_rmse) > 1
            else 0.0
        )
        assert result.std_metrics["rmse"] == pytest.approx(
            expected_std, rel=1e-6, abs=1e-9,
        )
    # Metadata records seeds, per-seed summaries and provenance.
    assert result.metadata["seeds"] == list(DEFAULT_EVAL_SEEDS)
    assert result.metadata["n_seeds"] == 5
    assert result.metadata["aggregation"] == "mean_of_per_seed_means"
    assert set(per_seed_means.keys()) == {
        str(int(s)) for s in DEFAULT_EVAL_SEEDS
    }
    assert "pooled_fold_mean_metrics" in result.metadata
    assert "provenance" in result.metadata


# --------------------------------------------------------------------------
# biomevae.utils (seeding + provenance)
# --------------------------------------------------------------------------


def test_set_global_seed_returns_reproducible_rng() -> None:
    rngs_a = set_global_seed(42)
    a1 = np.random.rand(4)
    rngs_b = set_global_seed(42)
    a2 = np.random.rand(4)
    assert rngs_a.seed == 42
    assert rngs_b.seed == 42
    assert np.allclose(a1, a2), "NumPy global state is not reproducible across calls"
    # The returned ``numpy_rng`` is an independent Generator that also
    # yields reproducible sequences.
    first = set_global_seed(7).numpy_rng.normal(size=3)
    second = set_global_seed(7).numpy_rng.normal(size=3)
    assert np.allclose(first, second)


def test_capture_provenance_has_expected_keys() -> None:
    record = capture_provenance(seeds=[42, 43, 44, 45, 46])
    assert "captured_at" in record
    assert "platform" in record and "python_version" in record["platform"]
    assert "packages" in record and "numpy" in record["packages"]
    assert "torch" in record
    assert record["seeds"] == [42, 43, 44, 45, 46]
