"""Reconstruction utilities and cross-validation helpers.

This module adds a non-negative matrix factorization (NMF) baseline together
with utilities to compare it against the existing neural-network-based models
in :mod:`biomevae`.  The cross-validation helpers rely on the Gabriel
bi-cross-validation holdout—also known as the "Gabriel split"—which withholds a
Cartesian product of rows and columns for validation.  This scheme offers a
structured alternative to Poisson thinning when evaluating unsupervised models
on tabular microbiome matrices (``[samples × features]``).

All returned metrics operate in the transformed space that was provided to the
estimator (for example, after a :func:`numpy.log1p` transform) so that the
numbers are directly comparable across methods.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.decomposition import NMF

from .data import load_matrix
from .trainers.train_loop import (
    train_once,
    train_once_philrvae,
    train_once_tree_dtm_vae,
    train_once_hyp_philrvae,
    train_once_dsvae,
)

__all__ = [
    "gabriel_split",
    "CrossValResult",
    "compute_reconstruction_metrics",
    "cross_validate_nmf",
    "cross_validate_nmf_multi_seed",
    "select_nmf_rank",
    "fit_nmf_embeddings",
    "cross_validate_vae",
    "cross_validate_vae_multi_seed",
    "compare_with_nmf",
    "compare_with_nmf_multi_seed",
    "compare_all_methods",
    "compare_all_methods_multi_seed",
    "merge_cross_val_results",
    "plot_benchmark_figure",
    "plot_ordination_grid",
    "plot_enterosignature_ordination_grid",
    "plot_enterosignature_agreement",
    "plot_enterosignature_comparison",
    "load_counts",
    "load_latent",
    "compute_ordinations",
    "compute_pairwise_metric_stats",
    "compute_pairwise_seed_stats",
    "adjust_pvalues_bh",
    "adjust_pvalues_bonferroni",
]


@dataclass
class CrossValResult:
    """Container for cross-validation summaries.

    Attributes
    ----------
    fold_metrics:
        Per-fold metric dictionaries.  When the result was pooled across
        several seeds by :func:`merge_cross_val_results`, this list
        concatenates *every* seed's folds (``n_splits * n_seeds``
        entries) so that downstream paired fold-level tests / violin
        plots still have access to the raw values.
    mean_metrics:
        When the result is single-seeded this is the mean across folds.
        When the result was pooled by :func:`merge_cross_val_results`
        this is instead the *mean of the per-seed means*, matching the
        Bouthillier et al. (2021) across-seed aggregation used by the
        classifier evaluation pipeline.
    std_metrics:
        Standard deviation matching the aggregation used by
        ``mean_metrics``.  For single-seed results this is the unbiased
        std across folds; for pooled results it is the unbiased std of
        the per-seed means (a.k.a. the across-seed standard deviation).
    metadata:
        Optional dictionary with estimator-specific metadata (for example,
        hyper-parameters actually used for fitting).  Pooled results
        additionally populate ``metadata['seeds']``,
        ``metadata['n_seeds']``, ``metadata['per_seed_mean_metrics']``,
        ``metadata['per_seed_std_metrics']``,
        ``metadata['aggregation']`` and ``metadata['provenance']``.
    """

    fold_metrics: List[Mapping[str, float]]
    mean_metrics: Mapping[str, float]
    std_metrics: Mapping[str, float]
    metadata: Mapping[str, object] | None = None


def _summarise_metrics(fold_metrics: List[Mapping[str, float]]) -> tuple[Dict[str, float], Dict[str, float]]:
    if not fold_metrics:
        raise ValueError("fold_metrics must contain at least one element")
    keys = set().union(*(m.keys() for m in fold_metrics))
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in sorted(keys):
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        mean[key] = float(values.mean())
        std[key] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _two_sided_sign_test_p_value(n_positive: int, n_negative: int) -> float:
    n = n_positive + n_negative
    if n == 0:
        return 1.0
    prob_obs = math.comb(n, n_positive) * (0.5 ** n)
    p_value = 0.0
    for i in range(n + 1):
        prob_i = math.comb(n, i) * (0.5 ** n)
        if prob_i <= prob_obs + 1e-12:
            p_value += prob_i
    return min(1.0, float(p_value))


def adjust_pvalues_bonferroni(p_values: List[float]) -> List[float]:
    """Apply Bonferroni correction to a list of p-values."""

    if not p_values:
        return []
    m = len(p_values)
    return [min(1.0, float(p) * m) for p in p_values]


def adjust_pvalues_bh(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction to a list of p-values."""

    if not p_values:
        return []
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    prev = 1.0
    for rank, (idx, p_value) in enumerate(reversed(indexed), start=1):
        k = m - rank + 1
        adj = min(prev, (m / k) * float(p_value))
        adjusted[idx] = min(1.0, adj)
        prev = adjusted[idx]
    return adjusted


def compute_pairwise_metric_stats(
    results: Mapping[str, CrossValResult],
    metric: str,
) -> List[Dict[str, float | int | str]]:
    """Compute pairwise sign-test comparisons for a fold-level metric.

    .. note::

        This function treats fold metrics as if they were independent
        samples and is therefore only meaningful when every model is
        evaluated on *identical* fold partitions.  For the repository's
        multi-seed protocol prefer :func:`compute_pairwise_seed_stats`,
        which operates on per-seed means (one sample per seed) and uses
        a paired Wilcoxon signed-rank test combined with the Nadeau &
        Bengio (2003) corrected t-test.
    """

    names = sorted(results.keys())
    comparisons: List[Dict[str, float | int | str]] = []
    for idx, name_a in enumerate(names):
        for name_b in names[idx + 1 :]:
            result_a = results[name_a]
            result_b = results[name_b]
            try:
                values_a = np.array(
                    [float(metrics[metric]) for metrics in result_a.fold_metrics],
                    dtype=float,
                )
                values_b = np.array(
                    [float(metrics[metric]) for metrics in result_b.fold_metrics],
                    dtype=float,
                )
            except KeyError as exc:
                raise ValueError(
                    f"Metric '{metric}' missing from fold metrics for comparison."
                ) from exc
            if values_a.size != values_b.size:
                raise ValueError(
                    "Fold counts do not match for "
                    f"{name_a!r} and {name_b!r} (metric '{metric}')."
                )
            diffs = values_a - values_b
            mean_diff = float(np.mean(diffs))
            median_diff = float(np.median(diffs))
            n_positive = int(np.sum(diffs > 0))
            n_negative = int(np.sum(diffs < 0))
            n_used = n_positive + n_negative
            p_value = _two_sided_sign_test_p_value(n_positive, n_negative)
            comparisons.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "n": n_used,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "p_value": p_value,
                }
            )
    return comparisons


def _extract_per_seed_means(
    result: CrossValResult, metric: str,
) -> tuple[List[int], List[float]]:
    """Return paired (seed, value) lists for ``metric`` from ``result``.

    Falls back gracefully when ``metadata['per_seed_mean_metrics']`` is
    missing by using the single-seed ``mean_metrics`` (treated as one
    pseudo-seed ``"0"``).  Seeds for which the metric is not present are
    skipped.
    """
    metadata = result.metadata or {}
    per_seed = metadata.get("per_seed_mean_metrics")
    if isinstance(per_seed, Mapping) and per_seed:
        seeds_out: List[int] = []
        values_out: List[float] = []
        for seed_key, metrics in per_seed.items():
            if not isinstance(metrics, Mapping):
                continue
            if metric not in metrics:
                continue
            try:
                seed_int = int(seed_key)
            except (TypeError, ValueError):
                continue
            try:
                seeds_out.append(seed_int)
                values_out.append(float(metrics[metric]))
            except (TypeError, ValueError):
                seeds_out.pop()
        return seeds_out, values_out
    # Single-seed fallback: pretend we have one pseudo-seed.
    if metric in result.mean_metrics:
        return [0], [float(result.mean_metrics[metric])]
    return [], []


def _paired_wilcoxon_p_value(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value with a SciPy fallback.

    Returns ``1.0`` when there are fewer than two non-zero differences.
    """
    nonzero = diffs[diffs != 0.0]
    if nonzero.size < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        stat = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return float(stat.pvalue)
    except Exception:  # pragma: no cover - SciPy edge cases
        # Fall back to the sign test on the same differences.
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        return _two_sided_sign_test_p_value(n_pos, n_neg)


def _nadeau_bengio_corrected_t_p_value(
    diffs: np.ndarray,
    *,
    train_fraction: float,
) -> float:
    """Two-sided Nadeau & Bengio (2003) corrected paired-t p-value.

    The standard paired t-test overstates significance when the same
    training set is reused across folds/seeds because the per-fold errors
    are correlated.  Nadeau & Bengio (2003) propose inflating the
    variance estimate by ``(1/n + rho/(1 - rho))`` where ``rho`` is the
    train/test ratio.  We use ``rho = (1 - train_fraction)`` as a
    reasonable proxy for the Gabriel-split holdout fraction used across
    the repository.  Returns ``1.0`` when ``diffs`` has fewer than two
    entries or zero variance.
    """
    n = int(diffs.size)
    if n < 2:
        return 1.0
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))
    if var_diff <= 0.0:
        return 1.0 if mean_diff == 0.0 else 0.0
    rho = max(0.0, min(0.999, 1.0 - float(train_fraction)))
    correction = 1.0 / n + rho / (1.0 - rho)
    corrected_var = var_diff * correction
    if corrected_var <= 0.0:
        return 1.0 if mean_diff == 0.0 else 0.0
    t_stat = mean_diff / math.sqrt(corrected_var)
    try:
        from scipy.stats import t as _t

        p_value = 2.0 * float(_t.sf(abs(t_stat), df=n - 1))
    except Exception:  # pragma: no cover - SciPy optional path
        # Normal approximation is a reasonable backstop for n ≈ 5.
        from math import erfc

        p_value = float(erfc(abs(t_stat) / math.sqrt(2.0)))
    return max(0.0, min(1.0, p_value))


def compute_pairwise_seed_stats(
    results: Mapping[str, CrossValResult],
    metric: str,
    *,
    train_fraction: float = 0.9,
) -> List[Dict[str, float | int | str]]:
    """Compute pairwise seed-level comparisons for ``metric``.

    Operates on per-seed means (one value per seed) rather than fold
    metrics so that the differences are genuinely independent samples.
    For each pair of methods this returns:

    * ``n`` — number of shared seeds,
    * ``mean_diff`` / ``median_diff`` — seed-level difference summaries,
    * ``p_value_sign`` — two-sided sign test,
    * ``p_value_wilcoxon`` — two-sided Wilcoxon signed-rank (SciPy),
    * ``p_value_tcorrected`` — Nadeau & Bengio corrected paired t-test,
    * ``p_value`` — ``p_value_tcorrected`` (used as the canonical value
      by downstream multiple-testing corrections).

    Missing seeds are intersected across the pair.  If the two methods
    share fewer than two seeds the comparison is still emitted with
    ``p_value = 1.0`` so that callers can render "insufficient data"
    cells rather than dropping methods silently.
    """
    names = sorted(results.keys())
    comparisons: List[Dict[str, float | int | str]] = []
    for idx, name_a in enumerate(names):
        for name_b in names[idx + 1 :]:
            seeds_a, vals_a = _extract_per_seed_means(results[name_a], metric)
            seeds_b, vals_b = _extract_per_seed_means(results[name_b], metric)
            lookup_a = dict(zip(seeds_a, vals_a))
            lookup_b = dict(zip(seeds_b, vals_b))
            shared = sorted(set(lookup_a.keys()) & set(lookup_b.keys()))
            if len(shared) < 2:
                comparisons.append(
                    {
                        "model_a": name_a,
                        "model_b": name_b,
                        "mean_diff": float("nan"),
                        "median_diff": float("nan"),
                        "n": len(shared),
                        "n_positive": 0,
                        "n_negative": 0,
                        "p_value_sign": 1.0,
                        "p_value_wilcoxon": 1.0,
                        "p_value_tcorrected": 1.0,
                        "p_value": 1.0,
                    }
                )
                continue
            paired_a = np.asarray([lookup_a[s] for s in shared], dtype=float)
            paired_b = np.asarray([lookup_b[s] for s in shared], dtype=float)
            diffs = paired_a - paired_b
            mean_diff = float(np.mean(diffs))
            median_diff = float(np.median(diffs))
            n_positive = int(np.sum(diffs > 0))
            n_negative = int(np.sum(diffs < 0))
            p_sign = _two_sided_sign_test_p_value(n_positive, n_negative)
            p_wil = _paired_wilcoxon_p_value(diffs)
            p_t = _nadeau_bengio_corrected_t_p_value(
                diffs, train_fraction=train_fraction,
            )
            comparisons.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "n": len(shared),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "p_value_sign": p_sign,
                    "p_value_wilcoxon": p_wil,
                    "p_value_tcorrected": p_t,
                    "p_value": p_t,
                }
            )
    return comparisons


def gabriel_split(
    X: np.ndarray,
    train_fraction: float = 0.9,
    *,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Gabriel-style holdout indices for bi-cross-validation.

    The returned tuple contains the indices for the training rows, the held-out
    validation rows, and the held-out validation columns respectively.  The
    validation set corresponds to the Cartesian product between the validation
    rows and columns, matching the Gabriel bi-cross-validation scheme described
    by Owen and Perry (2009).  ``train_fraction`` controls the expected
    proportion of matrix entries used for fitting.  The function ensures that at
    least one row and one column remain in both the training and validation
    splits.

    Parameters
    ----------
    X:
        Two-dimensional array containing the data to split.  Only the shape is
        used; the array is not modified.
    train_fraction:
        Fraction of entries that should remain for training.  Must lie in
        ``(0, 1)``.  A value of ``0.9`` means that roughly 10% of entries are
        reserved for validation.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    train_rows, val_rows, val_cols:
        Three sorted ``numpy.ndarray`` instances with the indices for the
        training rows, validation rows, and validation columns respectively.

    Raises
    ------
    ValueError
        If ``train_fraction`` is not in ``(0, 1)`` or if the input matrix has
        fewer than two rows or columns (Gabriel splitting is undefined in that
        case).
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")

    counts = np.asarray(X)
    if counts.ndim != 2:
        raise ValueError("Expected a 2D array")

    n_rows, n_cols = counts.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError("gabriel_split requires at least two rows and two columns")

    rng = np.random.default_rng(seed)

    holdout_fraction = 1.0 - train_fraction
    # Select the number of held-out rows/columns so that their Cartesian product
    # roughly matches the desired holdout fraction.  Use ``sqrt`` to spread the
    # holdout evenly between rows and columns and clamp to valid bounds.
    row_fraction = math.sqrt(holdout_fraction)
    col_fraction = math.sqrt(holdout_fraction)

    n_val_rows = int(round(row_fraction * n_rows))
    n_val_cols = int(round(col_fraction * n_cols))

    n_val_rows = max(1, min(n_val_rows, n_rows - 1))
    n_val_cols = max(1, min(n_val_cols, n_cols - 1))

    perm_rows = rng.permutation(n_rows)
    perm_cols = rng.permutation(n_cols)

    val_rows = np.sort(perm_rows[:n_val_rows])
    train_rows = np.sort(perm_rows[n_val_rows:])
    val_cols = np.sort(perm_cols[:n_val_cols])

    return train_rows, val_rows, val_cols


def compute_reconstruction_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive set of reconstruction diagnostics.

    Besides the classic mean absolute error (``mae``), mean squared error
    (``mse``), root mean squared error (``rmse``), and coefficient of
    determination (``r2``), the returned mapping now exposes several auxiliary
    statistics that are useful when diagnosing unexpectedly negative
    :math:`R^2` scores:

    ``target_mean``
        Mean of the validation targets.
    ``target_var``
        Average squared deviation of the targets around ``target_mean``.
    ``target_total_var``
        Sum of squared deviations across all validation entries.  This matches
        the denominator used in the :math:`R^2` computation.
    ``residual``
        Sum of squared reconstruction errors, i.e. the numerator of the
        :math:`R^2` expression.

    ``r2`` is reported as ``nan`` when the variance of ``target`` is zero.
    """

    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape")
    diff = prediction - target
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))

    target_mean = float(np.mean(target))
    centered = target - target_mean
    target_total_var = float(np.sum(np.square(centered)))
    target_var = float(np.mean(np.square(centered)))
    residual = float(np.sum(np.square(diff)))
    r2 = float("nan") if target_total_var == 0.0 else 1.0 - (residual / target_total_var)

    # Per-feature R² averaged across features.  The global R² above can be
    # misleadingly high when feature scales vary widely (common in microbiome
    # count data).  Per-feature R² gives equal weight to each feature.
    if target.ndim == 2 and target.shape[1] > 1:
        feat_ss_res = np.sum(np.square(diff), axis=0)          # (p,)
        feat_ss_tot = np.sum(np.square(target - target.mean(axis=0)), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            feat_r2 = 1.0 - feat_ss_res / feat_ss_tot
        valid = np.isfinite(feat_r2)
        r2_per_feature = float(np.mean(feat_r2[valid])) if np.any(valid) else float("nan")
    else:
        r2_per_feature = r2

    # Sparsity-aware metrics: precision/recall on zero vs nonzero entries.
    target_flat = target.ravel()
    pred_flat = prediction.ravel()
    true_zero = target_flat == 0.0
    true_nonzero = ~true_zero
    pred_zero = pred_flat == 0.0
    pred_nonzero = ~pred_zero

    n_total = len(target_flat)
    n_true_zero = int(true_zero.sum())
    n_true_nonzero = int(true_nonzero.sum())
    sparsity = n_true_zero / n_total if n_total > 0 else float("nan")

    # Zero precision: of entries predicted zero, how many are truly zero
    zero_precision = (
        float((true_zero & pred_zero).sum()) / float(pred_zero.sum())
        if pred_zero.any() else float("nan")
    )
    # Zero recall: of truly zero entries, how many are predicted zero
    zero_recall = (
        float((true_zero & pred_zero).sum()) / float(true_zero.sum())
        if true_zero.any() else float("nan")
    )
    # Nonzero precision: of entries predicted nonzero, how many are truly nonzero
    nonzero_precision = (
        float((true_nonzero & pred_nonzero).sum()) / float(pred_nonzero.sum())
        if pred_nonzero.any() else float("nan")
    )
    # Nonzero recall: of truly nonzero entries, how many are predicted nonzero
    nonzero_recall = (
        float((true_nonzero & pred_nonzero).sum()) / float(true_nonzero.sum())
        if true_nonzero.any() else float("nan")
    )

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "r2_per_feature": r2_per_feature,
        "residual": residual,
        "target_mean": target_mean,
        "target_var": target_var,
        "target_total_var": target_total_var,
        "sparsity": sparsity,
        "zero_precision": zero_precision,
        "zero_recall": zero_recall,
        "nonzero_precision": nonzero_precision,
        "nonzero_recall": nonzero_recall,
    }


def cross_validate_nmf(
    X_counts: np.ndarray,
    *,
    n_components: int,
    n_splits: int = 5,
    train_fraction: float = 0.9,
    log1p: bool = True,
    nmf_kwargs: Optional[MutableMapping[str, float | int | str]] = None,
    random_state: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
) -> CrossValResult:
    """Evaluate NMF reconstruction error using Gabriel bi-cross-validation."""

    if n_components <= 0:
        raise ValueError("n_components must be a positive integer")
    rng = np.random.default_rng(random_state)
    nmf_kwargs = {} if nmf_kwargs is None else dict(nmf_kwargs)

    fold_metrics: List[Mapping[str, float]] = []
    val_rows_count: Optional[int] = None
    val_cols_count: Optional[int] = None
    eval_As: Dict[str, np.ndarray] | None = None
    if taxonomy_eval:
        eval_As = {}
        n_features = X_counts.shape[1]
        for level, matrix in taxonomy_eval.items():
            A = np.asarray(matrix, dtype=np.float32)
            if A.ndim != 2:
                raise ValueError(
                    f"Taxonomy matrix for level '{level}' must be two-dimensional"
                )
            if A.shape[1] != n_features:
                raise ValueError(
                    "Taxonomy aggregation matrices must match the feature count of the input matrix."
                )
            eval_As[level] = A
    for fold in range(n_splits):
        fold_seed = int(rng.integers(0, 2**32 - 1))
        train_rows, val_rows, val_cols = gabriel_split(X_counts, train_fraction, seed=fold_seed)
        val_rows_count = val_rows.size
        val_cols_count = val_cols.size
        train_counts = X_counts[train_rows]
        val_counts = X_counts[val_rows]
        train_data = np.log1p(train_counts).astype(np.float32) if log1p else train_counts.astype(np.float32)
        val_data = np.log1p(val_counts).astype(np.float32) if log1p else val_counts.astype(np.float32)

        nmf_args = dict(nmf_kwargs)
        init = nmf_args.pop("init", "nndsvda")
        max_iter = int(nmf_args.pop("max_iter", 2000))
        nmf_args.setdefault("random_state", fold_seed)
        nmf = NMF(
            n_components=n_components,
            init=init,
            max_iter=max_iter,
            **nmf_args,
        )
        W_train = nmf.fit_transform(train_data)
        _ = W_train  # retained for clarity; reconstruction uses learned components

        # Project validation rows using only train columns so that
        # val_cols are truly held out during inference, mirroring the
        # column-masked encoding applied to the VAE.
        train_cols = np.setdiff1d(np.arange(X_counts.shape[1]), val_cols)
        H_train_cols = nmf.components_[:, train_cols]
        W_val = np.zeros(
            (val_data.shape[0], nmf.n_components_), dtype=val_data.dtype
        )
        val_train_cols = val_data[:, train_cols]
        for i in range(val_data.shape[0]):
            W_val[i], _ = nnls(H_train_cols.T, val_train_cols[i])
        recon_val = np.matmul(W_val, nmf.components_)
        target_block = val_data[:, val_cols]
        recon_block = recon_val[:, val_cols]
        metrics = compute_reconstruction_metrics(target_block, recon_block)
        if eval_As:
            for level, A_full in eval_As.items():
                A_subset = A_full[:, val_cols]
                active_groups = np.sum(A_subset, axis=1) > 0
                if not np.any(active_groups):
                    continue
                A_reduced = A_subset[active_groups]
                aggregated_target = target_block @ A_reduced.T
                aggregated_recon = recon_block @ A_reduced.T
                agg_metrics = compute_reconstruction_metrics(
                    aggregated_target, aggregated_recon
                )
                metrics[f"mae_tax_{level}"] = agg_metrics["mae"]
                metrics[f"rmse_tax_{level}"] = agg_metrics["rmse"]
                metrics[f"r2_tax_{level}"] = agg_metrics["r2"]
        fold_metrics.append(metrics)

    mean_metrics, std_metrics = _summarise_metrics(fold_metrics)
    metadata = {
        "n_components": n_components,
        "log1p": log1p,
        "train_fraction": train_fraction,
        "val_rows": int(val_rows_count) if val_rows_count is not None else 0,
        "val_cols": int(val_cols_count) if val_cols_count is not None else 0,
    }
    if nmf_kwargs:
        metadata["nmf_kwargs"] = dict(nmf_kwargs)
    metadata["random_state"] = None if random_state is None else int(random_state)
    if eval_As:
        metadata["taxonomy_levels"] = sorted(eval_As.keys())
        metadata["taxonomy_groups"] = {level: int(A.shape[0]) for level, A in eval_As.items()}
    return CrossValResult(fold_metrics, mean_metrics, std_metrics, metadata)


def _with_metadata(result: CrossValResult, updates: Mapping[str, object]) -> CrossValResult:
    metadata = dict(result.metadata) if result.metadata is not None else {}
    metadata.update(updates)
    return CrossValResult(
        fold_metrics=list(result.fold_metrics),
        mean_metrics=dict(result.mean_metrics),
        std_metrics=dict(result.std_metrics),
        metadata=metadata,
    )


def select_nmf_rank(
    X_counts: np.ndarray,
    *,
    candidates: List[int] | np.ndarray,
    n_splits: int = 5,
    train_fraction: float = 0.9,
    log1p: bool = True,
    nmf_kwargs: Optional[MutableMapping[str, float | int | str]] = None,
    random_state: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
    selection_metric: str = "rmse",
) -> CrossValResult:
    """Select the best NMF rank by minimizing the chosen bi-cross-validation metric."""

    if candidates is None:
        raise ValueError("candidates must be provided for NMF rank selection")
    unique_candidates = sorted({int(value) for value in candidates})
    if not unique_candidates or any(value <= 0 for value in unique_candidates):
        raise ValueError("candidates must contain positive integers")

    results: Dict[int, CrossValResult] = {}
    for n_components in unique_candidates:
        results[n_components] = cross_validate_nmf(
            X_counts,
            n_components=n_components,
            n_splits=n_splits,
            train_fraction=train_fraction,
            log1p=log1p,
            nmf_kwargs=nmf_kwargs,
            random_state=random_state,
            taxonomy_eval=taxonomy_eval,
        )
        if selection_metric not in results[n_components].mean_metrics:
            raise KeyError(
                f"Metric '{selection_metric}' not available for NMF rank {n_components}"
            )

    best_rank = min(
        unique_candidates,
        key=lambda n: float(results[n].mean_metrics[selection_metric]),
    )
    best_result = results[best_rank]
    rank_scores = {
        str(n): float(results[n].mean_metrics[selection_metric]) for n in unique_candidates
    }
    return _with_metadata(
        best_result,
        {
            "selected_rank": int(best_rank),
            "rank_candidates": list(unique_candidates),
            "selection_metric": selection_metric,
            "rank_scores": rank_scores,
        },
    )


def fit_nmf_embeddings(
    X_counts: np.ndarray,
    *,
    n_components: int,
    log1p: bool = True,
    nmf_kwargs: Mapping[str, float | int | str] | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Fit an NMF model on the full dataset and return the sample embeddings."""

    if n_components <= 0:
        raise ValueError("n_components must be a positive integer")

    counts = np.asarray(X_counts)
    if counts.ndim != 2:
        raise ValueError("Input matrix must be two-dimensional")

    data = np.log1p(counts).astype(np.float32) if log1p else counts.astype(np.float32)

    nmf_args = dict(nmf_kwargs) if nmf_kwargs is not None else {}
    init = nmf_args.pop("init", "nndsvda")
    max_iter = int(nmf_args.pop("max_iter", 2000))
    nmf_args.setdefault("random_state", random_state)

    nmf = NMF(
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        **nmf_args,
    )
    embeddings = nmf.fit_transform(data)
    return embeddings.astype(np.float32, copy=False)


def _prepare_params(params: Mapping[str, object]) -> Dict[str, object]:
    cfg = dict(params)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("val_split", 0.1)
    cfg.setdefault("early_stop", 50)
    cfg.setdefault("model_type", "euclid")
    cfg.setdefault("model_kwargs", {})
    cfg.setdefault("epochs", 400)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("latent_dim", 16)
    cfg.setdefault("hidden", [256, 128, 64])
    cfg.setdefault("activation", "leakyrelu")
    cfg.setdefault("layer_norm", False)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("lr", 2e-3)
    cfg.setdefault("optimizer", "adam")
    cfg.setdefault("weight_decay", 0.0)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("log1p", False)
    cfg.setdefault("standardize", False)
    cfg.setdefault("objective", "beta")
    cfg.setdefault("recon", "mse")
    cfg.setdefault("huber_delta", 1.0)
    cfg.setdefault("kl_warmup", 300)
    cfg.setdefault("beta_max", 0.05)
    cfg.setdefault("free_bits", 0.0)
    cfg.setdefault("capacity_start", 0.0)
    cfg.setdefault("capacity_end", None)
    cfg.setdefault("capacity_epochs", 160)
    cfg.setdefault("capacity_gamma", 1.0)
    cfg.setdefault("tax_levels", [])
    cfg.setdefault("tax_loss_weight", 0.0)
    cfg.setdefault("tax_As", None)
    cfg.setdefault("lap_L", None)
    cfg.setdefault("lap_weight", 0.0)
    if isinstance(cfg.get("model_kwargs"), dict):
        cfg["model_kwargs"] = dict(cfg["model_kwargs"])
    if isinstance(cfg.get("hidden"), tuple):
        cfg["hidden"] = list(cfg["hidden"])
    return cfg


def _prepare_philr_cv_data(
    X_counts: np.ndarray,
    cfg: Dict[str, object],
) -> Dict[str, object]:
    """Build tree-aligned data for the PhILR family cross-validation.

    Returns a dict with ``X_leaf`` (samples x tree leaves in
    ``taxg.leaf_ids`` order) and ``taxg`` (the TaxonomyGraph).
    """
    from pathlib import Path

    from .models.taxonomy_tree import build_taxonomy_graph_from_phyla_tsv

    taxonomy_path = cfg["taxonomy_path"]
    model_kwargs = cfg.get("model_kwargs", {}) or {}
    keep_prefixes = bool(model_kwargs.get("keep_prefixes", False))
    has_header = bool(model_kwargs.get("taxonomy_has_header", False))
    taxg = build_taxonomy_graph_from_phyla_tsv(
        Path(taxonomy_path),
        keep_prefixes=keep_prefixes,
        has_header=has_header,
        on_duplicate_leaf="ignore_same",
    )

    feature_clades = cfg.get("feature_clades")
    n_leaves = len(taxg.leaf_ids)
    n_samples = X_counts.shape[0]

    if feature_clades is None:
        X_leaf = X_counts[:, :n_leaves].astype(np.float32)
    else:
        input_path = cfg.get("input_path")
        if input_path:
            from .taxonomy import load_feature_clades
            sgb_features = load_feature_clades(input_path)
        else:
            sgb_features = list(feature_clades)
        sgb_name_to_col = {name: i for i, name in enumerate(sgb_features)}
        X_leaf = np.zeros((n_samples, n_leaves), dtype=np.float32)
        for li, nid in enumerate(taxg.leaf_ids):
            col = sgb_name_to_col.get(taxg.node_names[nid])
            if col is not None:
                X_leaf[:, li] = X_counts[:, col].astype(np.float32)

    return {"X_leaf": X_leaf.astype(np.float32), "taxg": taxg}


def _prepare_tree_dmt_cv_data(
    X_counts: np.ndarray,
    cfg: Dict[str, object],
) -> Dict[str, object]:
    """Build tree-aligned data for TreeDTM-VAE cross-validation.

    Reorders ``X_counts`` to the taxonomy tree's leaf order and aggregates
    leaf values to the full tree-node tensor consumed by the model.
    """
    from pathlib import Path

    from .models.taxonomy_tree import (
        aggregate_leaf_matrix_to_nodes,
        build_taxonomy_graph_from_phyla_tsv,
    )
    from .models.tree_dtm_vae import build_tree_topology

    taxonomy_path = cfg["taxonomy_path"]
    model_kwargs = cfg.get("model_kwargs", {}) or {}
    keep_prefixes = bool(model_kwargs.get("keep_prefixes", False))
    has_header = bool(model_kwargs.get("taxonomy_has_header", False))
    taxg = build_taxonomy_graph_from_phyla_tsv(
        Path(taxonomy_path),
        keep_prefixes=keep_prefixes,
        has_header=has_header,
        on_duplicate_leaf="ignore_same",
    )
    topo = build_tree_topology(taxg)

    feature_clades = cfg.get("feature_clades")
    n_samples = X_counts.shape[0]
    n_leaves = topo.n_leaves

    if feature_clades is None:
        X_leaf = X_counts[:, :n_leaves].astype(np.float32)
    else:
        input_path = cfg.get("input_path")
        if input_path:
            from .taxonomy import load_feature_clades
            sgb_features = load_feature_clades(input_path)
        else:
            sgb_features = list(feature_clades)
        sgb_name_to_col = {name: i for i, name in enumerate(sgb_features)}
        X_leaf = np.zeros((n_samples, n_leaves), dtype=np.float32)
        for li, nid in enumerate(taxg.leaf_ids):
            col = sgb_name_to_col.get(taxg.node_names[nid])
            if col is not None:
                X_leaf[:, li] = X_counts[:, col].astype(np.float32)

    X_nodes = aggregate_leaf_matrix_to_nodes(taxg, X_leaf)
    return {
        "X_leaves": X_leaf.astype(np.float32),
        "X_nodes": X_nodes.astype(np.float32),
        "topo": topo,
        "taxg": taxg,
    }


def cross_validate_vae(
    X_counts: np.ndarray,
    *,
    params: Mapping[str, object],
    n_splits: int = 5,
    train_fraction: float = 0.9,
    seed: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
) -> CrossValResult:
    """Cross-validate a VAE-type model using Gabriel-style splitting."""

    cfg = _prepare_params(params)
    model_type = str(cfg.get("model_type", "euclid"))

    # Compositional-likelihood models (philrvae / tree-dtm-vae and their
    # Gaussian counterparts) operate on raw-count space and handle their
    # own transforms internally. The variable name ``is_nb_model`` is
    # retained for backward compatibility with downstream callers; it
    # now reads "is a count-space-output model whose reconstruction must
    # be log1p-mapped before metric computation".
    is_nb_model = model_type in (
        "philrvae",
        "tree-dtm-vae",
        "hyperbolic-philrvae",
        "dsvae",
    )
    if is_nb_model:
        log1p_flag = False
        do_standardize = False
        cfg["standardize"] = False
    else:
        log1p_flag = bool(cfg.get("log1p", False))
        do_standardize = bool(cfg.get("standardize", False))
        # Let train_once() handle standardization internally.  With
        # external_val the scaler is fitted on all Gabriel training rows
        # (there is no redundant inner split).
        cfg["standardize"] = do_standardize

    # For TreeDTM-VAE, build the tree-aligned dataset once up front.
    tree_dmt_data = None
    if model_type == "tree-dtm-vae":
        tree_dmt_data = _prepare_tree_dmt_cv_data(X_counts, cfg)

    # For the new PhILR family, build a TaxonomyGraph + reorder X_counts to
    # the model's leaf order once up front.
    philr_data = None
    if model_type in ("philrvae", "hyperbolic-philrvae"):
        philr_data = _prepare_philr_cv_data(X_counts, cfg)

    rng = np.random.default_rng(seed)
    eval_As: Dict[str, np.ndarray] | None = None
    # Use the actual feature count (may differ for tree-dtm-vae).
    if tree_dmt_data:
        cv_data = tree_dmt_data["X_leaves"]
    elif philr_data:
        cv_data = philr_data["X_leaf"]
    else:
        cv_data = X_counts
    if taxonomy_eval:
        eval_As = {}
        n_features = cv_data.shape[1]
        for level, matrix in taxonomy_eval.items():
            A = np.asarray(matrix, dtype=np.float32)
            if A.ndim != 2:
                raise ValueError(
                    f"Taxonomy matrix for level '{level}' must be two-dimensional"
                )
            if A.shape[1] != n_features:
                # Taxonomy matrices are aligned to X_counts, skip if
                # feature count doesn't match (tree-dtm-vae reordering).
                continue
            eval_As[level] = A
        if not eval_As:
            eval_As = None

    fold_metrics: List[Mapping[str, float]] = []

    val_rows_count: Optional[int] = None
    val_cols_count: Optional[int] = None
    for fold in range(n_splits):
        fold_seed = int(rng.integers(0, 2**32 - 1))
        train_rows, val_rows, val_cols = gabriel_split(cv_data, train_fraction, seed=fold_seed)
        val_rows_count = val_rows.size
        val_cols_count = val_cols.size
        sample_names = [f"sample_{i}" for i in train_rows]

        tmpdir = tempfile.mkdtemp(prefix="biomevae_cv_")
        try:
            if model_type == "philrvae":
                taxg = philr_data["taxg"]
                X_leaf = philr_data["X_leaf"]
                res = train_once_philrvae(
                    X_leaf[train_rows].astype(np.float32),
                    sample_names, tmpdir, cfg, taxg,
                    seed=fold_seed, verbose=False, return_model=True,
                    external_val_leaf=X_leaf[val_rows].astype(np.float32),
                )
            elif model_type == "tree-dtm-vae":
                X_nodes = tree_dmt_data["X_nodes"]
                X_leaves = tree_dmt_data["X_leaves"]
                topo = tree_dmt_data["topo"]
                res = train_once_tree_dtm_vae(
                    X_nodes[train_rows], X_leaves[train_rows],
                    sample_names, tmpdir, cfg, topo,
                    seed=fold_seed, verbose=False, return_model=True,
                    external_val_nodes=X_nodes[val_rows],
                    external_val_leaves=X_leaves[val_rows],
                )
            elif model_type == "hyperbolic-philrvae":
                taxg = philr_data["taxg"]
                X_leaf = philr_data["X_leaf"]
                res = train_once_hyp_philrvae(
                    X_leaf[train_rows].astype(np.float32),
                    sample_names, tmpdir, cfg, taxg,
                    seed=fold_seed, verbose=False, return_model=True,
                    external_val_leaf=X_leaf[val_rows].astype(np.float32),
                )
            elif model_type == "dsvae":
                # DSVAE's reconstruction path is identical across the
                # supervised and unsupervised variants (same PhILR encoder
                # + NB decoder); only the classifier + SupCon heads differ.
                # During Gabriel CV we measure count-space reconstruction
                # only, so force ``supervised=False`` to avoid pulling
                # sample metadata into this CLI's contract — the
                # reconstruction capacity of the backbone is what the
                # benchmark is comparing.
                dsvae_cfg = dict(cfg)
                dsvae_cfg["supervised"] = False
                train_counts = cv_data[train_rows].astype(np.float32)
                val_counts_raw = cv_data[val_rows].astype(np.float32)
                res = train_once_dsvae(
                    train_counts, sample_names, tmpdir, dsvae_cfg,
                    seed=fold_seed, verbose=False, return_model=True,
                    external_val=val_counts_raw,
                )
            else:
                train_counts = cv_data[train_rows]
                val_counts = cv_data[val_rows]
                train_data = np.log1p(train_counts).astype(np.float32) if log1p_flag else train_counts.astype(np.float32)
                val_data_es = np.log1p(val_counts).astype(np.float32) if log1p_flag else val_counts.astype(np.float32)
                # Pass Gabriel validation rows as the external early-stopping
                # set so that train_once() uses *all* of train_data for
                # gradient updates instead of creating a redundant inner split.
                res = train_once(
                    train_data, sample_names, tmpdir, cfg,
                    seed=fold_seed, verbose=False, return_model=True,
                    external_val=val_data_es,
                )

            model = res["model"]
            if model is None:
                raise RuntimeError("train function did not return a model")

            device = torch.device(cfg.get("device", "cpu"))
            model = model.to(device)
            model.eval()

            if model_type in ("philrvae", "hyperbolic-philrvae"):
                # New PhILR family: input is samples x leaves in tree leaf
                # order; encode returns (mu, logvar) on the latent (tangent
                # space for the hyperbolic variant); decode returns
                # {coord_mu, leaf_prob}. Scale leaf_prob by the masked
                # library size to put the prediction back in count space.
                val_leaf = cv_data[val_rows].astype(np.float32)
                val_masked = val_leaf.copy()
                val_masked[:, val_cols] = 0.0
                data_kind = cfg.get("data_kind", "relative")
                with torch.no_grad():
                    tensor = torch.from_numpy(val_masked).to(device)
                    mu_val, _ = model.encode(tensor, data_kind=data_kind)
                    dec = model.decode(mu_val)
                    leaf_prob = dec["leaf_prob"]
                    lib = tensor.sum(dim=1, keepdim=True).clamp(min=1.0)
                    recon = (leaf_prob * lib).cpu().numpy()
                target = np.log1p(val_leaf[:, val_cols])

            elif model_type == "dsvae":
                # DSVAE still uses the legacy TreeSpec PhILR + NB decoder.
                val_raw = cv_data[val_rows].astype(np.float32)
                val_masked = val_raw.copy()
                val_masked[:, val_cols] = 0.0
                with torch.no_grad():
                    tensor = torch.from_numpy(val_masked).to(device)
                    mu_val, _ = model.encode(tensor)
                    lib = tensor.sum(dim=1, keepdim=True).clamp(min=1.0)
                    dec_out = model.decode(mu_val, lib)
                    if isinstance(dec_out, tuple):
                        mu_x, logit_pi = dec_out
                        pi = torch.sigmoid(logit_pi)
                        recon = ((1.0 - pi) * mu_x).cpu().numpy()
                    else:
                        recon = dec_out.cpu().numpy()
                target = np.log1p(val_raw[:, val_cols])

            elif model_type == "tree-dtm-vae":
                from .models.taxonomy_tree import aggregate_leaf_matrix_to_nodes

                taxg = tree_dmt_data["taxg"]
                X_leaves = tree_dmt_data["X_leaves"]
                val_leaves = X_leaves[val_rows].astype(np.float32)

                # Mask the held-out leaf columns and re-aggregate to node
                # values so the encoder cannot peek at the holdout block.
                val_leaves_masked = val_leaves.copy()
                val_leaves_masked[:, val_cols] = 0.0
                val_nodes_masked = aggregate_leaf_matrix_to_nodes(taxg, val_leaves_masked)

                with torch.no_grad():
                    t_nodes = torch.from_numpy(val_nodes_masked.astype(np.float32)).to(device)
                    mu_val, _ = model.encode(t_nodes)
                    leaf_prob = model.decode(mu_val)["leaf_prob"]
                    # Scale leaf probabilities by the (masked) library
                    # size to put the prediction in count space, matching
                    # the other count-output models.
                    lib = torch.from_numpy(
                        val_leaves_masked.sum(axis=1, keepdims=True).astype(np.float32)
                    ).to(device).clamp(min=1.0)
                    recon = (leaf_prob * lib).cpu().numpy()
                target = np.log1p(val_leaves[:, val_cols])

            else:
                # Standard VAE path (unchanged).
                val_counts = cv_data[val_rows]
                # Load the scaler that train_once() fitted on the training
                # rows (with external_val, all of train_data is used).
                scaler: Optional[Dict[str, np.ndarray]] = None
                scaler_path = os.path.join(tmpdir, "feature_scaler.npz")
                if do_standardize and os.path.exists(scaler_path):
                    npz = np.load(scaler_path)
                    scaler = {"mean": npz["mean"], "std": npz["std"]}

                val_data = np.log1p(val_counts).astype(np.float32) if log1p_flag else val_counts.astype(np.float32)
                if do_standardize and scaler is not None:
                    mean = scaler["mean"]
                    std = scaler["std"]
                    val_proc = (val_data - mean) / std
                else:
                    val_proc = val_data

                # Mask held-out columns before encoding so the encoder
                # cannot use features that belong to the Gabriel holdout
                # block.  This makes the column holdout effective during
                # inference, not only during evaluation.
                val_masked = val_proc.copy()
                val_masked[:, val_cols] = 0.0

                with torch.no_grad():
                    tensor = torch.from_numpy(val_masked).to(device)
                    mu_val, _ = model.encode(tensor)
                    recon_proc = model.decoder(mu_val).cpu().numpy()

                if do_standardize and scaler is not None:
                    recon = recon_proc * std + mean
                else:
                    recon = recon_proc
                target = val_data[:, val_cols]

            recon_block = recon[:, val_cols]
            if is_nb_model:
                recon_block = np.log1p(np.clip(recon_block, 0, None))
            metrics = compute_reconstruction_metrics(target, recon_block)
            if eval_As:
                for level, A_full in eval_As.items():
                    A_subset = A_full[:, val_cols]
                    active_groups = np.sum(A_subset, axis=1) > 0
                    if not np.any(active_groups):
                        continue
                    A_reduced = A_subset[active_groups]
                    aggregated_target = target @ A_reduced.T
                    aggregated_recon = recon_block @ A_reduced.T
                    agg_metrics = compute_reconstruction_metrics(
                        aggregated_target, aggregated_recon
                    )
                    metrics[f"mae_tax_{level}"] = agg_metrics["mae"]
                    metrics[f"rmse_tax_{level}"] = agg_metrics["rmse"]
                    metrics[f"r2_tax_{level}"] = agg_metrics["r2"]
            fold_metrics.append(metrics)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    mean_metrics, std_metrics = _summarise_metrics(fold_metrics)
    metadata = {
        "train_fraction": train_fraction,
        "log1p": log1p_flag,
        "standardize": do_standardize,
        "val_rows": int(val_rows_count) if val_rows_count is not None else 0,
        "val_cols": int(val_cols_count) if val_cols_count is not None else 0,
    }
    latent_dim = cfg.get("latent_dim")
    if latent_dim is not None:
        metadata["latent_dim"] = int(latent_dim)
    if eval_As:
        metadata["taxonomy_levels"] = sorted(eval_As.keys())
        metadata["taxonomy_groups"] = {level: int(A.shape[0]) for level, A in eval_As.items()}
    return CrossValResult(fold_metrics, mean_metrics, std_metrics, metadata)


def compare_with_nmf(
    X_counts: np.ndarray,
    *,
    method_name: str,
    vae_params: Mapping[str, object],
    nmf_components: int | None = None,
    nmf_rank_candidates: List[int] | None = None,
    nmf_selection_metric: str = "rmse",
    n_splits: int = 5,
    train_fraction: float = 0.9,
    seed: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, CrossValResult]:
    """Compare one neural model against the NMF baseline."""

    log1p_flag = bool(vae_params.get("log1p", True))
    if nmf_rank_candidates:
        nmf_result = select_nmf_rank(
            X_counts,
            candidates=nmf_rank_candidates,
            n_splits=n_splits,
            train_fraction=train_fraction,
            log1p=log1p_flag,
            random_state=seed,
            taxonomy_eval=taxonomy_eval,
            selection_metric=nmf_selection_metric,
        )
    else:
        if nmf_components is None:
            raise ValueError("nmf_components must be provided when rank candidates are omitted")
        nmf_result = cross_validate_nmf(
            X_counts,
            n_components=nmf_components,
            n_splits=n_splits,
            train_fraction=train_fraction,
            log1p=log1p_flag,
            random_state=seed,
            taxonomy_eval=taxonomy_eval,
        )
    vae_result = cross_validate_vae(
        X_counts,
        params=vae_params,
        n_splits=n_splits,
        train_fraction=train_fraction,
        seed=seed,
        taxonomy_eval=taxonomy_eval,
    )
    return {"nmf": nmf_result, method_name: vae_result}


def compare_all_methods(
    X_counts: np.ndarray,
    *,
    methods: Mapping[str, Mapping[str, object]],
    nmf_components: int | None = None,
    nmf_rank_candidates: List[int] | None = None,
    nmf_selection_metric: str = "rmse",
    n_splits: int = 5,
    train_fraction: float = 0.9,
    seed: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
    verbose: bool = False,
) -> Dict[str, CrossValResult]:
    """Evaluate all provided methods together with the NMF baseline."""

    if not methods:
        raise ValueError("methods cannot be empty")

    first_params = next(iter(methods.values()))
    log1p_flag = bool(first_params.get("log1p", True))
    if verbose:
        method_list = ", ".join(sorted(methods.keys()))
        print(
            "compare_all_methods: starting comparison "
            f"(methods=[{method_list}], log1p={log1p_flag}, "
            f"splits={n_splits}, train_fraction={train_fraction}, seed={seed})"
        )
    for name, params in methods.items():
        if bool(params.get("log1p", True)) != log1p_flag:
            raise ValueError(
                "All methods must agree on the log1p setting to ensure a fair comparison."
            )
    if nmf_rank_candidates:
        if verbose:
            print(
                "compare_all_methods: selecting NMF rank from candidates "
                f"{list(nmf_rank_candidates)} using metric '{nmf_selection_metric}'"
            )
        nmf_result = select_nmf_rank(
            X_counts,
            candidates=nmf_rank_candidates,
            n_splits=n_splits,
            train_fraction=train_fraction,
            log1p=log1p_flag,
            random_state=seed,
            taxonomy_eval=taxonomy_eval,
            selection_metric=nmf_selection_metric,
        )
        if verbose:
            selected = nmf_result.metadata.get("selected_rank") if nmf_result.metadata else None
            print(f"compare_all_methods: selected NMF rank {selected}")
    else:
        if nmf_components is None:
            raise ValueError("nmf_components must be provided when rank candidates are omitted")
        if verbose:
            print(f"compare_all_methods: running NMF baseline with {nmf_components} components")
        nmf_result = cross_validate_nmf(
            X_counts,
            n_components=nmf_components,
            n_splits=n_splits,
            train_fraction=train_fraction,
            log1p=log1p_flag,
            random_state=seed,
            taxonomy_eval=taxonomy_eval,
        )

    results: Dict[str, CrossValResult] = {"nmf": nmf_result}

    for name, params in methods.items():
        if verbose:
            print(f"compare_all_methods: cross-validating {name}")
        results[name] = cross_validate_vae(
            X_counts,
            params=params,
            n_splits=n_splits,
            train_fraction=train_fraction,
            seed=seed,
            taxonomy_eval=taxonomy_eval,
        )
        if verbose:
            print(f"compare_all_methods: finished {name}")
    if verbose:
        print("compare_all_methods: completed all methods")
    return results


# ---------------------------------------------------------------------------
# Multi-seed evaluation helpers
# ---------------------------------------------------------------------------


def _resolve_seeds(
    seeds: Optional[Iterable[int]] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """Resolve a seed specification into a concrete list of integer seeds.

    Mirrors :func:`biomevae.classify.normalise_seeds` but imported locally to
    avoid creating an import cycle between ``reconstruction`` and ``classify``.
    Defaults to the canonical 5 evaluation seeds ``(42, 43, 44, 45, 46)``.
    """
    from .classify import DEFAULT_EVAL_SEEDS

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


def _across_seed_summary(
    per_seed_mean: Dict[str, Dict[str, float]],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Compute the mean-of-means and unbiased std over the per-seed means.

    Each seed contributes one value per metric (the mean of that seed's
    folds).  The returned ``mean_metrics`` is the average of those
    per-seed means and ``std_metrics`` is their unbiased standard
    deviation.  Metrics that appear in some seeds but not others are
    averaged only over the seeds that provide them.  ``std`` is ``0.0``
    when fewer than two seeds contributed a finite value, matching the
    convention used by :func:`_summarise_metrics`.
    """
    if not per_seed_mean:
        return {}, {}
    all_keys = sorted({k for metrics in per_seed_mean.values() for k in metrics.keys()})
    mean_out: Dict[str, float] = {}
    std_out: Dict[str, float] = {}
    for key in all_keys:
        vals = [
            float(metrics[key])
            for metrics in per_seed_mean.values()
            if key in metrics and np.isfinite(float(metrics[key]))
        ]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean_out[key] = float(arr.mean())
        std_out[key] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean_out, std_out


def merge_cross_val_results(
    per_seed: Sequence[CrossValResult],
    seeds: Sequence[int],
) -> CrossValResult:
    """Pool several :class:`CrossValResult` instances (one per seed) into one.

    ``mean_metrics`` and ``std_metrics`` on the returned result are the
    across-seed mean and unbiased standard deviation of the per-seed
    means — i.e. each seed contributes exactly one sample per metric,
    which is the statistically principled way to aggregate repeated
    evaluations of the same learning pipeline (Bouthillier et al.,
    2021).  Fold-level metrics from every seed are still concatenated
    into ``fold_metrics`` for paired tests, violin plots, etc.

    ``metadata`` gains ``seeds``, ``n_seeds``, ``per_seed_mean_metrics``,
    ``per_seed_std_metrics``, ``pooled_fold_mean_metrics`` (the naive
    mean-of-all-folds kept for backwards compatibility with existing
    tooling) and a ``provenance`` block from
    :func:`biomevae.utils.capture_provenance`.
    """
    if not per_seed:
        raise ValueError("per_seed must contain at least one CrossValResult")
    if len(per_seed) != len(seeds):
        raise ValueError("per_seed and seeds must have the same length")

    pooled_fold_metrics: List[Mapping[str, float]] = []
    per_seed_mean: Dict[str, Dict[str, float]] = {}
    per_seed_std: Dict[str, Dict[str, float]] = {}
    for s, result in zip(seeds, per_seed):
        pooled_fold_metrics.extend(result.fold_metrics)
        per_seed_mean[str(int(s))] = dict(result.mean_metrics)
        per_seed_std[str(int(s))] = dict(result.std_metrics)

    mean_metrics, std_metrics = _across_seed_summary(per_seed_mean)

    # Keep the old fold-level mean around for consumers that want the
    # "pool every fold and take a plain mean" view (e.g. violin plots).
    pooled_fold_mean, pooled_fold_std = (
        _summarise_metrics(pooled_fold_metrics)
        if pooled_fold_metrics
        else ({}, {})
    )

    metadata: Dict[str, object] = (
        dict(per_seed[0].metadata) if per_seed[0].metadata is not None else {}
    )
    metadata["seeds"] = [int(s) for s in seeds]
    metadata["n_seeds"] = len(seeds)
    metadata["per_seed_mean_metrics"] = per_seed_mean
    metadata["per_seed_std_metrics"] = per_seed_std
    metadata["pooled_fold_mean_metrics"] = pooled_fold_mean
    metadata["pooled_fold_std_metrics"] = pooled_fold_std
    metadata["aggregation"] = "mean_of_per_seed_means"

    # Capture provenance once per merged result.  Import lazily to avoid
    # pulling in the utils package when reconstruction is used without
    # the rest of biomevae (e.g. in unit tests that stub torch).
    try:
        from .utils import capture_provenance

        metadata["provenance"] = capture_provenance(seeds=seeds)
    except Exception:  # pragma: no cover - best-effort provenance
        pass

    return CrossValResult(
        fold_metrics=pooled_fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        metadata=metadata,
    )


def cross_validate_nmf_multi_seed(
    X_counts: np.ndarray,
    *,
    n_components: int,
    n_splits: int = 5,
    train_fraction: float = 0.9,
    log1p: bool = True,
    nmf_kwargs: Optional[MutableMapping[str, float | int | str]] = None,
    seeds: Optional[Iterable[int]] = None,
    random_state: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
) -> CrossValResult:
    """Run :func:`cross_validate_nmf` once per seed and pool the results."""
    resolved = _resolve_seeds(seeds, random_state)
    per_seed = [
        cross_validate_nmf(
            X_counts,
            n_components=n_components,
            n_splits=n_splits,
            train_fraction=train_fraction,
            log1p=log1p,
            nmf_kwargs=nmf_kwargs,
            random_state=s,
            taxonomy_eval=taxonomy_eval,
        )
        for s in resolved
    ]
    return merge_cross_val_results(per_seed, resolved)


def cross_validate_vae_multi_seed(
    X_counts: np.ndarray,
    *,
    params: Mapping[str, object],
    n_splits: int = 5,
    train_fraction: float = 0.9,
    seeds: Optional[Iterable[int]] = None,
    seed: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
) -> CrossValResult:
    """Run :func:`cross_validate_vae` once per seed and pool the results."""
    resolved = _resolve_seeds(seeds, seed)
    per_seed = [
        cross_validate_vae(
            X_counts,
            params=params,
            n_splits=n_splits,
            train_fraction=train_fraction,
            seed=s,
            taxonomy_eval=taxonomy_eval,
        )
        for s in resolved
    ]
    return merge_cross_val_results(per_seed, resolved)


def compare_with_nmf_multi_seed(
    X_counts: np.ndarray,
    *,
    method_name: str,
    vae_params: Mapping[str, object],
    nmf_components: int | None = None,
    nmf_rank_candidates: List[int] | None = None,
    nmf_selection_metric: str = "rmse",
    n_splits: int = 5,
    train_fraction: float = 0.9,
    seeds: Optional[Iterable[int]] = None,
    seed: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, CrossValResult]:
    """Run :func:`compare_with_nmf` once per seed and pool per-method results."""
    resolved = _resolve_seeds(seeds, seed)
    per_seed_results: List[Dict[str, CrossValResult]] = [
        compare_with_nmf(
            X_counts,
            method_name=method_name,
            vae_params=vae_params,
            nmf_components=nmf_components,
            nmf_rank_candidates=nmf_rank_candidates,
            nmf_selection_metric=nmf_selection_metric,
            n_splits=n_splits,
            train_fraction=train_fraction,
            seed=s,
            taxonomy_eval=taxonomy_eval,
        )
        for s in resolved
    ]
    merged: Dict[str, CrossValResult] = {}
    method_names = list(per_seed_results[0].keys())
    for name in method_names:
        merged[name] = merge_cross_val_results(
            [ps[name] for ps in per_seed_results], resolved,
        )
    return merged


def compare_all_methods_multi_seed(
    X_counts: np.ndarray,
    *,
    methods: Mapping[str, Mapping[str, object]],
    nmf_components: int | None = None,
    nmf_rank_candidates: List[int] | None = None,
    nmf_selection_metric: str = "rmse",
    n_splits: int = 5,
    train_fraction: float = 0.9,
    seeds: Optional[Iterable[int]] = None,
    seed: Optional[int] = None,
    taxonomy_eval: Mapping[str, np.ndarray] | None = None,
    verbose: bool = False,
) -> Dict[str, CrossValResult]:
    """Run :func:`compare_all_methods` once per seed and pool per-method results.

    Default seeds are :data:`biomevae.classify.DEFAULT_EVAL_SEEDS`.  Setting
    ``seeds=[42]`` recovers the previous single-seed behaviour.
    """
    resolved = _resolve_seeds(seeds, seed)
    if verbose:
        seed_list = ", ".join(str(s) for s in resolved)
        print(
            f"compare_all_methods_multi_seed: running across {len(resolved)} "
            f"seeds [{seed_list}]"
        )
    per_seed_results: List[Dict[str, CrossValResult]] = []
    for idx, s in enumerate(resolved):
        if verbose:
            print(
                f"compare_all_methods_multi_seed: seed {s} "
                f"({idx + 1}/{len(resolved)})"
            )
        per_seed_results.append(
            compare_all_methods(
                X_counts,
                methods=methods,
                nmf_components=nmf_components,
                nmf_rank_candidates=nmf_rank_candidates,
                nmf_selection_metric=nmf_selection_metric,
                n_splits=n_splits,
                train_fraction=train_fraction,
                seed=s,
                taxonomy_eval=taxonomy_eval,
                verbose=verbose,
            )
        )
    merged: Dict[str, CrossValResult] = {}
    method_names = list(per_seed_results[0].keys())
    for name in method_names:
        merged[name] = merge_cross_val_results(
            [ps[name] for ps in per_seed_results], resolved,
        )
    return merged


def plot_benchmark_figure(
    results: Mapping[str, CrossValResult],
    metric: str = "rmse",
    *,
    title: str | None = None,
    baseline: str = "nmf",
    figsize: tuple[float, float] = (7.0, 4.0),
    output: str | None = None,
) -> tuple["matplotlib.figure.Figure", np.ndarray]:
    """Create a publication-ready bar chart summarising benchmark results.

    Parameters
    ----------
    results:
        Mapping from method names to :class:`CrossValResult` objects produced by
        :func:`compare_all_methods` or :func:`compare_with_nmf`.
    metric:
        Evaluation metric to plot.  Must be present in ``mean_metrics`` for
        every result; defaults to ``"rmse"``.
    title:
        Optional title rendered above the axes.
    baseline:
        Method name that should be highlighted as the baseline (defaults to
        ``"nmf"``).  If provided, the baseline bar is rendered first using a
        neutral colour.
    figsize:
        Size of the created figure passed directly to
        :func:`matplotlib.pyplot.subplots`.
    output:
        Optional path where the resulting figure will be saved using
        :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    figure, axes:
        Tuple with the created :class:`matplotlib.figure.Figure` and the
        corresponding :class:`matplotlib.axes.Axes` instance to enable further
        customisation by callers.

    Notes
    -----
    The helper keeps external dependencies lightweight by importing
    :mod:`matplotlib` on demand.  Install the package with the ``figure``
    extra to ensure the plotting dependency is available.
    """

    if not results:
        raise ValueError("results cannot be empty")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "matplotlib is required to generate benchmark figures; install"
            " biomevae with the 'figure' extra or add matplotlib to your"
            " environment"
        ) from exc

    items: list[tuple[str, float, float]] = []
    for name, summary in results.items():
        if metric not in summary.mean_metrics:
            raise KeyError(f"Metric '{metric}' not available for method '{name}'")
        mean_value = float(summary.mean_metrics[metric])
        std_value = float(summary.std_metrics.get(metric, 0.0))
        items.append((name, mean_value, std_value))
    items.sort(key=lambda item: item[1])

    names = [item[0] for item in items]
    means = [item[1] for item in items]
    stds = [item[2] for item in items]

    figure, bar_ax = plt.subplots(1, 1, figsize=figsize)
    if isinstance(bar_ax, np.ndarray):  # pragma: no cover - defensive guard
        bar_ax = bar_ax.flatten()[0]

    indices = np.arange(len(names))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    palette_iter = iter(palette)
    colours: list[str] = []
    for name in names:
        if baseline and name == baseline:
            colours.append("#6c757d")  # muted neutral tone for the baseline
        else:
            try:
                colour = next(palette_iter)
            except StopIteration:  # pragma: no cover - unlikely to trigger
                palette_iter = iter(palette)
                colour = next(palette_iter, "C0")
            colours.append(colour)

    bars = bar_ax.bar(indices, means, yerr=stds, capsize=6, color=colours, alpha=0.9)
    bar_ax.set_xticks(indices)
    display_labels: list[str] = []
    for name, summary in zip(names, (results[n] for n in names)):
        metadata = summary.metadata or {}
        dim: int | None = None
        if "latent_dim" in metadata:
            dim = int(metadata["latent_dim"])
        elif "selected_rank" in metadata:
            dim = int(metadata["selected_rank"])
        elif "n_components" in metadata:
            dim = int(metadata["n_components"])
        label = f"{name}\n(dim={dim})" if dim is not None else name
        display_labels.append(label)
    bar_ax.set_xticklabels(display_labels, rotation=15, ha="right")

    metric_label = metric.upper() if metric.lower() != "r2" else "R²"
    bar_ax.set_ylabel(metric_label)
    bar_ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins="auto", prune="lower"))
    bar_ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

    if title:
        bar_ax.set_title(title)

    bar_ax.margins(y=0.1)

    if metric.lower() not in {"mae", "rmse"}:
        for bar, mean, std in zip(bars, means, stds):
            if np.isfinite(mean):
                if abs(mean) >= 1:
                    value_fmt = "{:.2f}"
                else:
                    value_fmt = "{:.3f}"
                label = value_fmt.format(mean)
                if std > 0:
                    label = f"{label} ± {value_fmt.format(std)}"
                bar_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                )

    figure.tight_layout()

    if output is not None:
        figure.savefig(output, dpi=300, bbox_inches="tight")
        _base, _ext = os.path.splitext(str(output))
        if _ext.lower() != ".png":
            figure.savefig(_base + ".png", dpi=300, bbox_inches="tight")

    return figure, np.array([bar_ax])


def plot_ordination_grid(
    ordinations: Mapping[str, Mapping[str, np.ndarray]],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    output: str | None = None,
    order: list[str] | None = None,
    latent_dims: Mapping[str, int] | None = None,
) -> tuple["matplotlib.figure.Figure", np.ndarray]:
    """Render a grid of PCA/t-SNE ordinations for multiple embeddings."""

    if not ordinations:
        raise ValueError("ordinations cannot be empty")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "matplotlib is required to generate ordination figures; install"
            " biomevae with the 'figure' extra or add matplotlib to your"
            " environment"
        ) from exc

    labels = list(ordinations.keys())
    if order:
        ordered = [label for label in order if label in ordinations]
        remainder = [label for label in labels if label not in ordered]
        labels = ordered + remainder
    n_rows = len(labels)

    # Detect whether any space includes UMAP ordinations.
    has_umap = any("umap" in ordinations[label] for label in labels)
    n_cols = 3 if has_umap else 2
    if figsize is None:
        width = 10.5 if has_umap else 7.0
        figsize = (width, max(3.0, 3.0 * n_rows))

    figure, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row_idx, label in enumerate(labels):
        space = ordinations[label]
        try:
            pca_coords = np.asarray(space["pca"], dtype=float)
            tsne_coords = np.asarray(space["tsne"], dtype=float)
        except KeyError as exc:
            raise KeyError(f"Missing ordination '{exc.args[0]}' for space '{label}'") from exc
        umap_coords = np.asarray(space["umap"], dtype=float) if "umap" in space else None

        coords_list = [pca_coords, tsne_coords]
        if has_umap:
            coords_list.append(umap_coords)

        for col_idx, coords in enumerate(coords_list):
            axis = axes[row_idx, col_idx]
            if coords is None:
                axis.set_visible(False)
                continue
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(
                    f"Ordination for '{label}' must be a 2D array with at least two columns"
                )
            if coords.shape[1] > 2:
                coords = coords[:, :2]
            axis.scatter(
                coords[:, 0],
                coords[:, 1],
                s=18,
                alpha=0.8,
                edgecolors="none",
                color="#495057",
            )
            axis.grid(alpha=0.3, linestyle="--", linewidth=0.6)

        axes[row_idx, 0].set_title("PCA")
        axes[row_idx, 0].set_xlabel("PC1")
        axes[row_idx, 0].set_ylabel("PC2")

        axes[row_idx, 1].set_title("t-SNE")
        axes[row_idx, 1].set_xlabel("Dim 1")
        axes[row_idx, 1].set_ylabel("Dim 2")

        if has_umap:
            axes[row_idx, 2].set_title("UMAP")
            axes[row_idx, 2].set_xlabel("Dim 1")
            axes[row_idx, 2].set_ylabel("Dim 2")

        row_label = label.replace("_", " ")
        if latent_dims and label in latent_dims:
            row_label = f"{row_label}\n(dim={latent_dims[label]})"
        axes[row_idx, 0].text(
            0.02,
            0.98,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, linewidth=0.0),
        )

    if title:
        figure.suptitle(title)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        figure.tight_layout()

    if output is not None:
        figure.savefig(output, dpi=300, bbox_inches="tight")
        _base, _ext = os.path.splitext(str(output))
        if _ext.lower() != ".png":
            figure.savefig(_base + ".png", dpi=300, bbox_inches="tight")

    return figure, axes


def plot_enterosignature_ordination_grid(
    ordinations: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    *,
    signature_weights: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    output: str | None = None,
    order: list[str] | None = None,
    latent_dims: Mapping[str, int] | None = None,
) -> tuple["matplotlib.figure.Figure", np.ndarray]:
    """Render PCA/t-SNE ordinations with points colored by enterosignature clusters."""

    if not ordinations:
        raise ValueError("ordinations cannot be empty")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if signature_weights is not None and signature_weights.shape[0] != labels.shape[0]:
        raise ValueError("signature_weights must have the same number of rows as labels")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    palette = {label: cmap(idx % cmap.N) for idx, label in enumerate(unique_labels)}
    point_colors = np.array([palette[label] for label in labels])

    ordered_labels = list(ordinations.keys())
    if order:
        priority = [name for name in order if name in ordinations]
        remainder = [name for name in ordered_labels if name not in priority]
        ordered_labels = priority + remainder

    n_rows = len(ordered_labels)
    n_cols = 2
    if figsize is None:
        figsize = (7.5, max(3.0, 3.0 * n_rows))

    figure, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row_idx, label in enumerate(ordered_labels):
        space = ordinations[label]
        if "pca" not in space or "tsne" not in space:
            raise KeyError(f"Ordination for '{label}' must contain 'pca' and 'tsne' keys")
        pca_coords = np.asarray(space["pca"], dtype=float)
        tsne_coords = np.asarray(space["tsne"], dtype=float)

        for coords, axis in zip((pca_coords, tsne_coords), axes[row_idx]):
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(
                    f"Ordination for '{label}' must be a 2D array with at least two columns"
                )
            if coords.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Ordination '{label}' has {coords.shape[0]} samples, but labels have {labels.shape[0]}"
                )
            if coords.shape[1] > 2:
                coords = coords[:, :2]
            axis.scatter(
                coords[:, 0],
                coords[:, 1],
                s=22,
                alpha=0.85,
                edgecolors="none",
                c=point_colors,
            )
            axis.grid(alpha=0.3, linestyle="--", linewidth=0.6)

        axes[row_idx, 0].set_title("PCA")
        axes[row_idx, 0].set_xlabel("PC1")
        axes[row_idx, 0].set_ylabel("PC2")

        axes[row_idx, 1].set_title("t-SNE")
        axes[row_idx, 1].set_xlabel("Dim 1")
        axes[row_idx, 1].set_ylabel("Dim 2")

        row_label = label.replace("_", " ")
        if latent_dims and label in latent_dims:
            row_label = f"{row_label}\n(dim={latent_dims[label]})"
        axes[row_idx, 0].text(
            0.02,
            0.98,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, linewidth=0.0),
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=str(label), markerfacecolor=palette[label], markersize=8)
        for label in unique_labels
    ]
    legend = axes[0, 1].legend(
        handles=legend_handles,
        title="Cluster",
        loc="upper right",
        frameon=True,
    )
    if signature_weights is not None:
        weights_norm = signature_weights.astype(float, copy=True)
        row_sums = weights_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        weights_norm /= row_sums

        n_components = weights_norm.shape[1]
        cluster_composition = np.zeros((len(unique_labels), n_components), dtype=float)
        for idx, cluster_label in enumerate(unique_labels):
            members = weights_norm[labels == cluster_label]
            if members.size == 0:
                continue
            cluster_composition[idx] = members.mean(axis=0)

        sig_cmap = plt.get_cmap("tab20")
        sig_colors = [sig_cmap(idx % sig_cmap.N) for idx in range(n_components)]
        y_start = 0.62
        y_step = 0.18
        pie_size = 0.18
        x_start = 0.78
        for idx, cluster_label in enumerate(unique_labels):
            y_pos = y_start - idx * y_step
            if y_pos < 0.02:
                break
            inset_axis = axes[0, 1].inset_axes([x_start, y_pos, pie_size, pie_size])
            inset_axis.pie(
                cluster_composition[idx],
                colors=sig_colors,
                startangle=90,
                counterclock=False,
            )
            inset_axis.set_aspect("equal")
            inset_axis.set_xticks([])
            inset_axis.set_yticks([])
            inset_axis.set_title(f"C{cluster_label}", fontsize=7, pad=1)
        legend.set_zorder(10)

    if title:
        figure.suptitle(title)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        figure.tight_layout()

    if output is not None:
        figure.savefig(output, dpi=300, bbox_inches="tight")
        _base, _ext = os.path.splitext(str(output))
        if _ext.lower() != ".png":
            figure.savefig(_base + ".png", dpi=300, bbox_inches="tight")

    return figure, axes


def plot_enterosignature_comparison(
    compositions: Mapping[str, np.ndarray],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    output: str | None = None,
    enterosignature_labels: list[str] | None = None,
) -> tuple["matplotlib.figure.Figure", np.ndarray]:
    """Plot enterosignature mixture composition by embedding cluster."""

    if not compositions:
        raise ValueError("compositions cannot be empty")

    import matplotlib.pyplot as plt

    labels = list(compositions.keys())
    first_matrix = compositions[labels[0]]
    if first_matrix.ndim != 2:
        raise ValueError("composition matrices must be two-dimensional")
    n_clusters, n_components = first_matrix.shape
    for label, matrix in compositions.items():
        if matrix.shape != (n_clusters, n_components):
            raise ValueError(
                "All composition matrices must share the same shape; "
                f"'{label}' has shape {matrix.shape}, expected {(n_clusters, n_components)}."
            )

    if enterosignature_labels is None:
        enterosignature_labels = [f"Signature {idx + 1}" for idx in range(n_components)]
    if len(enterosignature_labels) != n_components:
        raise ValueError("enterosignature_labels length must match composition columns")

    n_embeddings = len(labels)
    ncols = min(2, n_embeddings)
    nrows = int(np.ceil(n_embeddings / ncols))
    if figsize is None:
        figsize = (max(6.0, 3.5 * ncols), max(3.6, 3.2 * nrows))

    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab20")
    colors = cmap.colors if hasattr(cmap, "colors") else None
    legend_handles = []

    for axis_idx, (axis, label) in enumerate(zip(axes.flat, labels)):
        matrix = compositions[label] * 100.0
        x = np.arange(n_clusters)
        bottom = np.zeros(n_clusters)
        for comp_idx in range(n_components):
            color = colors[comp_idx % len(colors)] if colors else None
            bars = axis.bar(
                x,
                matrix[:, comp_idx],
                bottom=bottom,
                color=color,
                label=enterosignature_labels[comp_idx],
            )
            if axis_idx == 0:
                legend_handles.append(bars[0])
            bottom += matrix[:, comp_idx]
        axis.set_title(label)
        axis.set_xticks(x)
        axis.set_xticklabels([str(idx) for idx in range(n_clusters)])
        axis.set_xlabel("Cluster")
        axis.set_ylabel("Percent of mixture")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)

    for axis in axes.flat[len(labels) :]:
        axis.axis("off")

    if title:
        figure.suptitle(title)

    figure.legend(
        handles=legend_handles,
        labels=enterosignature_labels,
        loc="upper center",
        ncol=min(4, n_components),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if output is not None:
        figure.savefig(output, dpi=300, bbox_inches="tight")
        _base, _ext = os.path.splitext(str(output))
        if _ext.lower() != ".png":
            figure.savefig(_base + ".png", dpi=300, bbox_inches="tight")

    return figure, axes


def plot_enterosignature_agreement(
    agreement: Mapping[str, float],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    output: str | None = None,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Plot adjusted Rand index agreement between embeddings and enterosignatures."""

    if not agreement:
        raise ValueError("agreement cannot be empty")

    import matplotlib.pyplot as plt

    labels = list(agreement.keys())
    scores = np.array([agreement[label] for label in labels], dtype=float)
    order = np.argsort(scores)[::-1]
    labels = [labels[idx] for idx in order]
    scores = scores[order]

    if figsize is None:
        figsize = (max(6.0, 0.7 * len(labels)), 3.6)

    figure, axis = plt.subplots(figsize=figsize)
    bars = axis.bar(range(len(labels)), scores, color="tab:blue", alpha=0.85)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Adjusted Rand index")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    if title:
        axis.set_title(title)
    for bar, score in zip(bars, scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()

    if output is not None:
        figure.savefig(output, dpi=300, bbox_inches="tight")
        _base, _ext = os.path.splitext(str(output))
        if _ext.lower() != ".png":
            figure.savefig(_base + ".png", dpi=300, bbox_inches="tight")

    return figure, axis


def load_counts(path: str, *, log1p: bool = False) -> np.ndarray:
    """Convenience wrapper to load the counts matrix from disk.

    This function mirrors :func:`biomevae.data.load_matrix` but returns just the
    matrix (discarding sample names) because the comparison utilities only
    require the numerical data.
    """

    X, _ = load_matrix(path, log1p=log1p)
    return X


def load_latent(path: str) -> np.ndarray:
    """Load a latent-space embedding matrix from disk.

    The helper expects a tab-delimited file where the first column contains
    sample identifiers and the remaining columns encode the latent
    representation (for example, the ``embeddings.tsv`` files written by the
    training CLI).  The identifiers are ignored—the return value focuses solely
    on the numeric embedding values in ``[samples × dimensions]`` format.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "pandas is required to load latent embeddings; install biomevae with the "
            "'figure' extra or add pandas to your environment"
        ) from exc

    df = pd.read_csv(path, sep="\t", index_col=0)
    if df.empty:
        raise ValueError(f"Latent embedding file '{path}' does not contain any samples")
    return df.to_numpy(dtype=np.float32, copy=False)


def compute_ordinations(
    X: np.ndarray,
    *,
    log1p: bool = True,
    random_state: int = 0,
    perplexity: float = 30.0,
    latent: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Compute PCA and t-SNE projections for visualising benchmark data.

    When ``latent`` embeddings are supplied the function returns ordinations for
    both the original matrix and the latent representation so that downstream
    plots can contrast how the learned space reshapes the data geometry.
    """

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "scikit-learn is required to compute PCA and t-SNE ordinations; "
            "install biomevae with the 'optuna' extra or add scikit-learn to your environment"
        ) from exc

    counts = np.asarray(X)
    if counts.ndim != 2:
        raise ValueError("Input matrix must be two-dimensional")

    latent_matrix: np.ndarray | None = None
    if latent is not None:
        latent_matrix = np.asarray(latent)
        if latent_matrix.ndim != 2:
            raise ValueError("Latent embeddings must be a two-dimensional array")
        if latent_matrix.shape[0] != counts.shape[0]:
            raise ValueError(
                "Latent embeddings must contain the same number of samples as the original matrix"
            )

    def _prepare(matrix: np.ndarray, apply_log1p: bool) -> np.ndarray:
        processed = np.log1p(matrix).astype(np.float32) if apply_log1p else matrix.astype(np.float32)
        if processed.shape[0] < 2:
            raise ValueError("At least two samples are required for ordination plots")
        return processed

    # Try importing UMAP; it's optional.
    try:
        from umap import UMAP as _UMAP
        _has_umap = True
    except ImportError:
        _has_umap = False

    def _compute(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        n_samples = matrix.shape[0]
        max_components = min(2, matrix.shape[1], n_samples)
        if max_components == 0:
            raise ValueError("Input matrix does not contain enough information for ordination")
        pca = PCA(n_components=max_components, random_state=random_state)
        pca_coords = pca.fit_transform(matrix)
        if pca_coords.shape[1] < 2:
            pca_coords = np.pad(
                pca_coords,
                ((0, 0), (0, 2 - pca_coords.shape[1])),
                constant_values=0.0,
            )

        effective_perplexity = float(perplexity)
        upper_bound = max(1.0, (n_samples - 1) / 3.0)
        if effective_perplexity >= upper_bound:
            effective_perplexity = max(1.0, min(perplexity, upper_bound))
        if effective_perplexity >= n_samples:
            effective_perplexity = max(1.0, n_samples - 1.0)

        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            perplexity=effective_perplexity,
        )
        tsne_coords = tsne.fit_transform(matrix)

        umap_coords: np.ndarray | None = None
        if _has_umap:
            n_neighbors = min(15, n_samples - 1)
            if n_neighbors >= 2:
                reducer = _UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                )
                umap_coords = reducer.fit_transform(matrix).astype(np.float32)

        return pca_coords.astype(np.float32), tsne_coords.astype(np.float32), umap_coords

    spaces: Dict[str, np.ndarray] = {"original": _prepare(counts, log1p)}
    if latent_matrix is not None:
        spaces["latent"] = _prepare(latent_matrix, False)

    multiple_spaces = len(spaces) > 1
    ordinations: Dict[str, np.ndarray] = {}
    for space_name, matrix in spaces.items():
        pca_coords, tsne_coords, umap_coords = _compute(matrix)
        suffix = f"_{space_name}" if multiple_spaces else ""
        ordinations[f"pca{suffix}"] = pca_coords
        ordinations[f"tsne{suffix}"] = tsne_coords
        if umap_coords is not None:
            ordinations[f"umap{suffix}"] = umap_coords

    return ordinations
