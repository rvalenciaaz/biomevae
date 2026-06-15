from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.metrics import pairwise_distances

from .reconstruction import compute_reconstruction_metrics
from .taxonomy import load_taxonomy_table

__all__ = [
    "EnterosignatureResult",
    "aggregate_genus_abundance",
    "bray_curtis_distances",
    "k_medoids",
    "compute_enterosignatures",
    "load_latent_embeddings",
    "cluster_embeddings",
]


@dataclass
class EnterosignatureResult:
    labels: np.ndarray
    sample_names: list[str]
    genus_abundance: np.ndarray
    genus_names: list[str]
    weights: np.ndarray
    basis: np.ndarray
    n_components: int
    n_clusters: int
    alpha: float
    median_cosine_similarity: Dict[int, float] = field(default_factory=dict)
    median_r2_score: Dict[int, float] = field(default_factory=dict)
    cosine_similarity_scores: Dict[int, list[float]] = field(default_factory=dict)
    r2_scores: Dict[int, list[float]] = field(default_factory=dict)
    mean_explained_variance: Dict[float, float] = field(default_factory=dict)


_UNASSIGNED_TOKENS = {
    "",
    "na",
    "nan",
    "none",
    "unknown",
    "unassigned",
    "unclassified",
}


def _is_unassigned_domain(label: str) -> bool:
    cleaned = str(label).strip().lower()
    return (
        cleaned in _UNASSIGNED_TOKENS
        or cleaned.startswith("na_")
        or "unassigned" in cleaned
        or "unclassified" in cleaned
        or cleaned.endswith("__")
    )


def _validate_matrix(matrix: np.ndarray, *, name: str) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")


def aggregate_genus_abundance(
    counts_path: str,
    taxonomy_path: str,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    raw = pd.read_csv(counts_path, sep="\t", dtype=str)
    if raw.shape[1] < 3:
        raise SystemExit(
            "Expected at least 3 columns: clade_name, NCBI_tax_id, and sample columns."
        )

    clade_names = raw.iloc[:, 0].astype(str)
    sample_cols = list(raw.columns[2:])
    abundances = raw.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    taxonomy = load_taxonomy_table(taxonomy_path)
    taxonomy_aligned = taxonomy.reindex(clade_names)
    if taxonomy_aligned.isna().any().any():
        taxonomy_aligned = taxonomy_aligned.fillna({lvl: f"NA_{lvl}" for lvl in taxonomy_aligned.columns})

    domain_mask = ~taxonomy_aligned["k"].astype(str).apply(_is_unassigned_domain)
    if not domain_mask.all():
        abundances = abundances.loc[domain_mask.values]
        taxonomy_aligned = taxonomy_aligned.loc[domain_mask]
        clade_names = clade_names.loc[domain_mask.values]
        if abundances.empty:
            raise SystemExit("All entries were removed after filtering unassigned domains.")

    genus_labels = taxonomy_aligned["g"].astype(str).to_numpy()
    abundances["genus"] = genus_labels
    genus_table = abundances.groupby("genus", sort=False).sum()

    genus_matrix = genus_table.to_numpy(dtype=np.float32).T
    genus_names = [str(name) for name in genus_table.index]

    if normalize:
        totals = genus_matrix.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        genus_matrix = genus_matrix / totals

    _validate_matrix(genus_matrix, name="Genus abundance matrix")
    return genus_matrix, sample_cols, genus_names


def bray_curtis_distances(matrix: np.ndarray) -> np.ndarray:
    _validate_matrix(matrix, name="Genus abundance matrix")
    distances = pairwise_distances(matrix, metric="braycurtis")
    return distances.astype(np.float32)


def k_medoids(
    distance_matrix: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    n_samples = distance_matrix.shape[0]
    if n_clusters < 1 or n_clusters > n_samples:
        raise ValueError("n_clusters must be between 1 and the number of samples")

    rng = np.random.RandomState(random_state)
    medoids = rng.choice(n_samples, n_clusters, replace=False)
    labels = np.full(n_samples, -1, dtype=int)

    for _ in range(max_iter):
        distances_to_medoids = distance_matrix[:, medoids]
        new_labels = np.argmin(distances_to_medoids, axis=1)

        new_medoids = medoids.copy()
        for cluster_idx in range(n_clusters):
            members = np.where(new_labels == cluster_idx)[0]
            if members.size == 0:
                available = np.setdiff1d(np.arange(n_samples), new_medoids)
                if available.size == 0:
                    continue
                nearest = distances_to_medoids[available].min(axis=1)
                new_medoids[cluster_idx] = available[int(np.argmax(nearest))]
                continue
            intra = distance_matrix[np.ix_(members, members)]
            medoid_local = int(np.argmin(intra.sum(axis=1)))
            new_medoids[cluster_idx] = members[medoid_local]

        if np.array_equal(new_medoids, medoids) and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        medoids = new_medoids
        labels = new_labels

    return labels, medoids


def _relative_sse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """1 - SSE / sum(observed²).  Not standard R²; uses total energy as reference."""
    denom = float(np.sum(observed ** 2))
    if denom == 0.0:
        return 0.0
    num = float(np.sum((observed - predicted) ** 2))
    return 1.0 - num / denom


def _cosine_similarity(observed: np.ndarray, predicted: np.ndarray) -> float:
    numerator = float(np.dot(observed, predicted))
    denom = float(np.linalg.norm(observed) * np.linalg.norm(predicted))
    if denom == 0.0:
        return 0.0
    return numerator / denom


def _fit_nmf(
    matrix: np.ndarray,
    n_components: int,
    *,
    alpha: float,
    random_state: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=alpha,
        alpha_H=alpha,
        l1_ratio=1.0,
    )
    weights = model.fit_transform(matrix)
    basis = model.components_
    return (
        weights.astype(matrix.dtype, copy=False),
        basis.astype(matrix.dtype, copy=False),
    )


def _bicross_validation_scores(
    matrix: np.ndarray,
    n_components: int,
    *,
    alpha: float,
    random_state: int,
    max_iter: int,
    repetitions: int,
    folds: int,
) -> tuple[list[float], list[float]]:
    _validate_matrix(matrix, name="Genus abundance matrix")
    if folds < 2:
        raise ValueError("bicross-validation folds must be at least 2")

    fold_root = int(round(math.sqrt(folds)))
    if fold_root * fold_root != folds:
        raise ValueError("bicross-validation folds must be a perfect square (e.g., 9)")

    n_rows, n_cols = matrix.shape
    if fold_root > n_rows or fold_root > n_cols:
        raise ValueError("bicross-validation folds exceed the matrix dimensions")

    rng = np.random.default_rng(random_state)
    explained_variances: list[float] = []
    cosine_similarities: list[float] = []

    for _rep in range(repetitions):
        row_perm = rng.permutation(n_rows)
        col_perm = rng.permutation(n_cols)
        row_groups = np.array_split(row_perm, fold_root)
        col_groups = np.array_split(col_perm, fold_root)

        for row_group in row_groups:
            if row_group.size == 0:
                continue
            val_rows = np.sort(row_group)
            train_rows = np.setdiff1d(np.arange(n_rows), val_rows)
            train_block = matrix[train_rows]
            val_block = matrix[val_rows]
            if not np.any(train_block) or not np.any(val_block):
                continue

            for col_group in col_groups:
                if col_group.size == 0:
                    continue
                val_cols = np.sort(col_group)
                fold_seed = int(rng.integers(0, 2**32 - 1))
                weights_train, basis_train = _fit_nmf(
                    train_block,
                    n_components,
                    alpha=alpha,
                    random_state=fold_seed,
                    max_iter=max_iter,
                )
                if not np.any(weights_train) or not np.any(basis_train):
                    continue

                weights_hold = non_negative_factorization(
                    val_block,
                    H=basis_train,
                    init="custom",
                    update_H=False,
                    solver="mu",
                    beta_loss="kullback-leibler",
                    max_iter=max_iter,
                    alpha_W=alpha,
                    alpha_H=alpha,
                    l1_ratio=1.0,
                    random_state=fold_seed,
                )[0]
                reconstructed = weights_hold @ basis_train
                validation_block = val_block[:, val_cols]
                reconstructed_block = reconstructed[:, val_cols]
                if validation_block.size == 0:
                    continue
                metrics = compute_reconstruction_metrics(
                    validation_block,
                    reconstructed_block,
                )
                r2 = metrics["r2"]
                if not np.isnan(r2):
                    explained_variances.append(r2)
                cosine_similarities.append(
                    _cosine_similarity(
                        validation_block.ravel(), reconstructed_block.ravel()
                    )
                )

    return explained_variances, cosine_similarities


def _select_n_components(
    matrix: np.ndarray,
    k_values: Sequence[int],
    *,
    random_state: int,
    max_iter: int,
    repetitions: int,
    folds: int,
    alpha: float,
) -> tuple[int, Dict[int, float], Dict[int, float], Dict[int, list[float]], Dict[int, list[float]]]:
    def _score_std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    median_cosine: Dict[int, float] = {}
    median_r2: Dict[int, float] = {}
    cosine_scores: Dict[int, list[float]] = {}
    r2_scores: Dict[int, list[float]] = {}
    for k in k_values:
        evs, cosine = _bicross_validation_scores(
            matrix,
            k,
            alpha=alpha,
            random_state=random_state,
            max_iter=max_iter,
            repetitions=repetitions,
            folds=folds,
        )
        cosine_scores[k] = cosine
        r2_scores[k] = evs
        median_cosine[k] = float(np.median(cosine)) if cosine else 0.0
        median_r2[k] = float(np.median(evs)) if evs else 0.0

    ordered = list(k_values)
    if len(ordered) == 1:
        return ordered[0], median_cosine, median_r2, cosine_scores, r2_scores

    max_cosine = max(median_cosine.values())
    best_cosine_k = min(
        k for k in ordered if np.isclose(median_cosine[k], max_cosine, rtol=1e-6, atol=1e-8)
    )
    cosine_threshold = max_cosine - _score_std(cosine_scores[best_cosine_k])
    cosine_candidates = [k for k in ordered if median_cosine[k] >= cosine_threshold]
    if len(cosine_candidates) == 1:
        best_k = cosine_candidates[0]
        return best_k, median_cosine, median_r2, cosine_scores, r2_scores

    max_r2 = max(median_r2[k] for k in cosine_candidates)
    best_r2_k = min(
        k for k in cosine_candidates if np.isclose(median_r2[k], max_r2, rtol=1e-6, atol=1e-8)
    )
    r2_threshold = max_r2 - _score_std(r2_scores[best_r2_k])
    r2_candidates = [k for k in cosine_candidates if median_r2[k] >= r2_threshold]
    best_k = min(r2_candidates)
    return best_k, median_cosine, median_r2, cosine_scores, r2_scores


def _select_alpha(
    matrix: np.ndarray,
    n_components: int,
    alpha_values: Sequence[float],
    *,
    random_state: int,
    max_iter: int,
    repetitions: int,
    folds: int,
) -> tuple[float, Dict[float, float]]:
    mean_ev: Dict[float, float] = {}
    ev_zero: list[float] = []

    for alpha in alpha_values:
        evs, _cos = _bicross_validation_scores(
            matrix,
            n_components,
            alpha=alpha,
            random_state=random_state,
            max_iter=max_iter,
            repetitions=repetitions,
            folds=folds,
        )
        mean_value = float(np.mean(evs)) if evs else 0.0
        mean_ev[float(alpha)] = mean_value
        if float(alpha) == 0.0:
            ev_zero = evs

    if not ev_zero:
        return float(alpha_values[0]), mean_ev

    mean_ev_zero = float(np.mean(ev_zero))
    std_ev_zero = float(np.std(ev_zero, ddof=1)) if len(ev_zero) > 1 else 0.0
    threshold = mean_ev_zero - std_ev_zero

    selected_alpha = float(alpha_values[-1])
    for alpha in reversed(alpha_values):
        if mean_ev[float(alpha)] >= threshold:
            selected_alpha = float(alpha)
            break

    return selected_alpha, mean_ev


def compute_enterosignatures(
    counts_path: str,
    taxonomy_path: str,
    n_components: int | None,
    n_clusters: int | None,
    *,
    random_state: int = 0,
    k_range: Sequence[int] | None = None,
    alpha_range: Sequence[float] | None = None,
    max_iter: int = 2000,
    bicross_folds: int = 9,
    bicross_repetitions: int = 100,
    cluster_max_iter: int = 300,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> EnterosignatureResult:
    genus_matrix, sample_names, genus_names = aggregate_genus_abundance(
        counts_path, taxonomy_path, normalize=True
    )
    if k_range is None:
        k_range = list(range(2, 11))
    if not k_range:
        raise ValueError("k_range must contain at least one candidate value.")
    k_range = sorted(set(k_range))
    if alpha_range is None:
        alpha_range = list(range(0, 101))
    if 0 not in alpha_range:
        alpha_range = list(alpha_range) + [0]
    alpha_range = sorted(set(alpha_range))

    total_steps = 3 + (1 if n_components is None else 0)
    current_step = 0

    def _report(message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(current_step, total_steps, message)

    _report("Starting enterosignature computation")

    if n_components is None:
        n_components, median_cosine, median_r2, cosine_scores, r2_scores = _select_n_components(
            genus_matrix,
            k_range,
            random_state=random_state,
            max_iter=max_iter,
            repetitions=bicross_repetitions,
            folds=bicross_folds,
            alpha=0.0,
        )
        current_step += 1
        _report("Selected k")
    else:
        median_cosine = {}
        median_r2 = {}
        cosine_scores = {}
        r2_scores = {}

    if n_clusters is None:
        n_clusters = n_components

    alpha, mean_ev = _select_alpha(
        genus_matrix,
        n_components,
        alpha_range,
        random_state=random_state,
        max_iter=max_iter,
        repetitions=bicross_repetitions,
        folds=bicross_folds,
    )
    current_step += 1
    _report("Selected alpha")

    weights, basis = _fit_nmf(
        genus_matrix,
        n_components,
        alpha=alpha,
        random_state=random_state,
        max_iter=max_iter,
    )
    current_step += 1
    _report("Fit NMF")
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights_norm = weights / row_sums
    labels = cluster_embeddings(
        weights_norm,
        n_clusters,
        random_state=random_state,
        max_iter=cluster_max_iter,
    )
    current_step += 1
    _report("Clustered samples")

    return EnterosignatureResult(
        labels=labels,
        sample_names=sample_names,
        genus_abundance=genus_matrix,
        genus_names=genus_names,
        weights=weights,
        basis=basis,
        n_components=n_components,
        n_clusters=n_clusters,
        alpha=alpha,
        median_cosine_similarity=median_cosine,
        median_r2_score=median_r2,
        cosine_similarity_scores=cosine_scores,
        r2_scores=r2_scores,
        mean_explained_variance=mean_ev,
    )


def load_latent_embeddings(
    path: str,
    *,
    sample_names: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path, sep="\t", index_col=0)
    if df.empty:
        raise ValueError(f"Latent embedding file '{path}' does not contain any samples")

    if sample_names is not None:
        missing = [name for name in sample_names if name not in df.index]
        if missing:
            raise ValueError(
                "Latent embedding file is missing samples: "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )
        df = df.loc[list(sample_names)]

    return df.to_numpy(dtype=np.float32, copy=False), list(df.index)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
    max_iter: int = 300,
) -> np.ndarray:
    _validate_matrix(embeddings, name="Embeddings")
    distances = pairwise_distances(embeddings, metric="euclidean")
    labels, _medoids = k_medoids(
        distances.astype(np.float32),
        n_clusters,
        random_state=random_state,
        max_iter=max_iter,
    )
    return labels
