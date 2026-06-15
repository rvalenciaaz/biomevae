"""CLI entry point to render benchmark figures with enterosignatures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
from scipy.spatial import procrustes
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    pairwise_distances,
    silhouette_score,
)

from biomevae.classify import DEFAULT_EVAL_SEEDS
from biomevae.enterosignatures import (
    cluster_embeddings,
    compute_enterosignatures,
    load_latent_embeddings,
)
from biomevae.reconstruction import (
    CrossValResult,
    cross_validate_nmf_multi_seed,
    load_counts,
    plot_benchmark_figure,
    plot_enterosignature_agreement,
    plot_enterosignature_comparison,
    plot_enterosignature_ordination_grid,
    plot_ordination_grid,
)
from biomevae.taxonomy import build_taxonomy_structures

from .benchmark_figure import (
    _collect_latent_dims,
    _load_results,
    _order_methods_by_metric,
    _parse_embedding_specs,
    _parse_figsize,
    _parse_renames,
    _print_pairwise_stats,
    _prepare_ordinations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("biomevae-benchmark-figures-enterosignatures")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more JSON files produced by reconstruction benchmarks",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        metavar="KEY",
        help=(
            "Metric key to visualise (repeatable). When omitted, all metrics shared "
            "across the methods are rendered."
        ),
    )
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument(
        "--baseline",
        default="nmf",
        help="Method name that should be highlighted as the baseline",
    )
    parser.add_argument(
        "--figsize",
        default="7x4",
        help="Figure size specified as WIDTHxHEIGHT in inches (default: 7x4)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path where the metric figures should be written",
    )
    parser.add_argument(
        "--rename",
        nargs="*",
        metavar="OLD=NEW",
        help="Rename method labels before plotting (repeatable)",
    )
    parser.add_argument(
        "--matrix",
        required=True,
        help="Counts matrix used to compute ordinations and enterosignatures",
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Taxonomy table containing genus assignments",
    )
    parser.add_argument(
        "--embedding",
        action="append",
        metavar="NAME=PATH",
        help="Latent embedding specification (repeatable; NAME must be unique)",
    )
    parser.add_argument(
        "--latent",
        default=None,
        help="Deprecated alias for a single embedding path (use --embedding NAME=PATH)",
    )
    parser.add_argument(
        "--matrix-log1p",
        dest="matrix_log1p",
        action="store_true",
        help="Apply log1p before computing ordinations (default)",
    )
    parser.add_argument(
        "--no-matrix-log1p",
        dest="matrix_log1p",
        action="store_false",
        help="Disable log1p transform before PCA/t-SNE",
    )
    parser.add_argument(
        "--ordinations-output",
        default=None,
        help="Optional path where the PCA/t-SNE grid should be written",
    )
    parser.add_argument(
        "--enterosignature-output",
        default=None,
        help="Optional path for the enterosignature-colored ordination grid",
    )
    parser.add_argument(
        "--enterosignature-title",
        default="Enterosignatures on PCA/t-SNE ordinations",
        help="Title for the enterosignature ordination grid",
    )
    parser.add_argument(
        "--signature-genus-output",
        default=None,
        help="Optional path for a TSV summary of genus contributions per enterosignature",
    )
    parser.add_argument(
        "--signature-genus-top",
        type=int,
        default=10,
        help="Number of top genera to display per enterosignature (default: 10).",
    )
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Optional path for the enterosignature composition plot",
    )
    parser.add_argument(
        "--comparison-title",
        default="Enterosignature composition by embedding cluster",
        help="Title for the enterosignature composition plot",
    )
    parser.add_argument(
        "--agreement-output",
        default=None,
        help="Optional path for the enterosignature agreement plot (ARI).",
    )
    parser.add_argument(
        "--agreement-title",
        default="Enterosignature agreement by embedding (adjusted Rand index)",
        help="Title for the enterosignature agreement plot",
    )
    parser.add_argument(
        "--rank-selection-output",
        default=None,
        help="Optional path for the enterosignature rank selection plot.",
    )
    parser.add_argument(
        "--geometry-output",
        default=None,
        help="Optional path for the Mantel-style distance matrix summary (TSV).",
    )
    parser.add_argument(
        "--geometry-plot-output",
        default=None,
        help="Optional path for Mantel distance scatter plot(s).",
    )
    parser.add_argument(
        "--mantel-permutations",
        type=int,
        default=999,
        help="Number of permutations for Mantel-style tests (default: 999).",
    )
    parser.add_argument(
        "--include-original-distance",
        action="store_true",
        help="Include original genus abundance distances in Mantel comparisons.",
    )
    parser.add_argument(
        "--procrustes-output",
        default=None,
        help="Optional path for Procrustes alignment plot(s).",
    )
    parser.add_argument(
        "--procrustes-stats-output",
        default=None,
        help="Optional path for Procrustes disparity summary (TSV).",
    )
    parser.add_argument(
        "--clustering-output",
        default=None,
        help="Optional path for clustering agreement summary (TSV).",
    )
    parser.add_argument(
        "--contingency-output",
        default=None,
        help="Optional path for ES vs VAE clustering contingency table(s) (TSV).",
    )
    parser.add_argument(
        "--contingency-plot-output",
        default=None,
        help="Optional path for ES vs VAE contingency heatmap plot(s).",
    )
    parser.add_argument(
        "--signatures",
        type=int,
        default=None,
        help=(
            "Optional override for the number of enterosignatures (NMF components). "
            "When omitted, bicross-validation selects the optimal k."
        ),
    )
    parser.add_argument(
        "--reuse-nmf-results",
        action="store_true",
        default=False,
        help=(
            "Reuse the NMF baseline metrics from the benchmark input instead of "
            "recomputing them with the selected enterosignature rank."
        ),
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help=(
            "Optional override for the number of mixture clusters. When omitted, the "
            "number of enterosignatures is used."
        ),
    )
    parser.add_argument(
        "--k-range",
        default="2-10",
        help="Inclusive range for k search, formatted as MIN-MAX (default: 2-10).",
    )
    parser.add_argument(
        "--alpha-range",
        default="0-100",
        help="Inclusive range for alpha search, formatted as MIN-MAX (default: 0-100).",
    )
    parser.add_argument(
        "--bicross-folds",
        type=int,
        default=9,
        help="Number of bicross-validation folds (perfect square, default: 9).",
    )
    parser.add_argument(
        "--bicross-repetitions",
        type=int,
        default=100,
        help="Number of bicross-validation repetitions (default: 100).",
    )
    parser.add_argument(
        "--nmf-max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for NMF fitting (default: 2000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help=(
            "Random seed for enterosignature bicross-validation, NMF fitting, and "
            "k-medoids initialisation (default: 0)"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVAL_SEEDS),
        help=(
            "Random seeds used when refreshing the NMF reconstruction baseline "
            "cross-validation (default: %(default)s). Fold metrics from every "
            "seed are pooled for reproducibility."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "DEPRECATED: legacy single-seed alias for the NMF reconstruction "
            "refresh. If provided, overrides --seeds with a single-seed evaluation."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum iterations for k-medoids clustering (default: 300)",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose logging (default)",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable verbose logging",
    )
    parser.set_defaults(matrix_log1p=True)
    parser.set_defaults(verbose=True)
    return parser


def _close_figure(figure) -> None:
    try:  # Ensure resources are released when running headless
        import matplotlib.pyplot as plt

        plt.close(figure)
    except ImportError:  # pragma: no cover - matplotlib already validated upstream
        pass


def _parse_range(value: str, *, name: str) -> tuple[int, int]:
    try:
        start_str, end_str = value.split("-", maxsplit=1)
        start = int(start_str)
        end = int(end_str)
    except ValueError as exc:
        raise SystemExit(f"{name} must be formatted as MIN-MAX, got '{value}'.") from exc
    if start > end:
        raise SystemExit(f"{name} must have MIN <= MAX, got '{value}'.")
    return start, end


def _format_path_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    return path / suffix


def _format_signature_genus_summary(
    *,
    basis,
    genus_names: list[str],
    top_n: int,
) -> list[str]:
    lines = []
    total_components = basis.shape[0]
    for component_idx in range(total_components):
        weights = basis[component_idx]
        top_indices = list(reversed(np.argsort(weights)))[:top_n]
        entries = ", ".join(
            f"{genus_names[idx]} ({weights[idx]:.4f})" for idx in top_indices
        )
        lines.append(f"Signature {component_idx + 1}: {entries}")
    return lines


def _extract_upper_triangle(distance_matrix: np.ndarray) -> np.ndarray:
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    upper = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    return upper.astype(float, copy=False)


def _mantel_test(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    permutations: int,
    random_state: int,
) -> tuple[float, float]:
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Mantel test requires distance matrices of the same shape")
    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    upper_a = _extract_upper_triangle(matrix_a)
    upper_b = _extract_upper_triangle(matrix_b)
    if upper_a.size == 0 or upper_b.size == 0:
        raise ValueError("Mantel test requires at least two samples")

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
        if denom == 0.0:
            return 0.0
        return float(np.dot(x_centered, y_centered) / denom)

    observed = _corr(upper_a, upper_b)
    if permutations == 0:
        return observed, 1.0

    rng = np.random.RandomState(random_state)
    permuted = np.empty(permutations, dtype=float)
    n = matrix_a.shape[0]
    for idx in range(permutations):
        order = rng.permutation(n)
        permuted_matrix = matrix_b[np.ix_(order, order)]
        permuted[idx] = _corr(upper_a, _extract_upper_triangle(permuted_matrix))

    extreme = np.sum(np.abs(permuted) >= abs(observed))
    p_value = (extreme + 1) / (permutations + 1)
    return observed, float(p_value)


def _compute_distance_matrix(matrix: np.ndarray, *, metric: str) -> np.ndarray:
    return pairwise_distances(matrix, metric=metric).astype(np.float32, copy=False)


def _compute_mds(distance_matrix: np.ndarray, *, random_state: int) -> np.ndarray:
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
    )
    return mds.fit_transform(distance_matrix)


def _write_tsv(path: Path, header: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _plot_procrustes(
    *,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    labels: list[str],
    title: str,
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - validated upstream
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords_a[:, 0], coords_a[:, 1], c="tab:blue", label=labels[0], alpha=0.8)
    ax.scatter(coords_b[:, 0], coords_b[:, 1], c="tab:orange", label=labels[1], alpha=0.8)
    for idx in range(coords_a.shape[0]):
        ax.plot(
            [coords_a[idx, 0], coords_b[idx, 0]],
            [coords_a[idx, 1], coords_b[idx, 1]],
            color="0.7",
            linewidth=0.5,
            zorder=0,
        )
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_distance_scatter(
    *,
    distances_a: np.ndarray,
    distances_b: np.ndarray,
    labels: tuple[str, str],
    title: str,
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - validated upstream
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(distances_a, distances_b, s=10, alpha=0.5)
    ax.set_xlabel(f"{labels[0]} distances")
    ax.set_ylabel(f"{labels[1]} distances")
    ax.set_title(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_contingency_heatmap(
    *,
    table: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - validated upstream
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(table, cmap="Blues")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Embedding cluster")
    ax.set_ylabel("Enterosignature cluster")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_selection(
    *,
    cosine_scores: Dict[int, list[float]],
    r2_scores: Dict[int, list[float]],
    selected_k: int,
    title: str,
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - validated upstream
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    if not cosine_scores:
        raise ValueError("cosine_scores cannot be empty")
    if not r2_scores:
        raise ValueError("r2_scores cannot be empty")
    if selected_k not in cosine_scores or selected_k not in r2_scores:
        raise ValueError("selected_k must be present in scores")

    ordered = sorted(cosine_scores)
    cosine_values = [
        cosine_scores[k] if cosine_scores[k] else [np.nan] for k in ordered
    ]
    r2_values = [r2_scores.get(k, []) or [np.nan] for k in ordered]
    positions = np.arange(1, len(ordered) + 1)
    selected_position = positions[ordered.index(selected_k)]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 6), sharex=True)
    cosine_ax, r2_ax = axes

    cosine_ax.boxplot(
        cosine_values,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": "tab:blue", "alpha": 0.2},
        medianprops={"color": "tab:blue", "linewidth": 1.5},
    )
    cosine_ax.axvline(selected_position, color="tab:red", linestyle="--", linewidth=1.2)
    cosine_ax.set_ylabel("Cosine similarity")
    cosine_ax.grid(alpha=0.3, linestyle="--", linewidth=0.6, axis="y")
    cosine_ax.set_title(title)

    r2_ax.boxplot(
        r2_values,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": "tab:green", "alpha": 0.2},
        medianprops={"color": "tab:green", "linewidth": 1.5},
    )
    r2_ax.axvline(selected_position, color="tab:red", linestyle="--", linewidth=1.2)
    r2_ax.set_xlabel("Rank (k)")
    r2_ax.set_ylabel("R\u00b2")
    r2_ax.grid(alpha=0.3, linestyle="--", linewidth=0.6, axis="y")

    r2_ax.set_xticks(positions)
    r2_ax.set_xticklabels([str(k) for k in ordered])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _refresh_nmf_baseline(
    results: Dict[str, CrossValResult],
    *,
    selected_rank: int,
    matrix_path: str,
    taxonomy_path: str,
    reuse: bool,
    verbose: bool,
    log: Callable[[str], None],
    seeds: Iterable[int],
) -> Dict[str, CrossValResult]:
    if reuse:
        log("Reusing NMF baseline metrics from the benchmark input.")
        return results
    if "nmf" not in results:
        log("No NMF baseline found in the benchmark input; skipping refresh.")
        return results
    if selected_rank <= 0:
        raise SystemExit("Selected enterosignature rank must be positive.")

    nmf_result = results["nmf"]
    metadata = nmf_result.metadata or {}
    existing_rank_raw = metadata.get("selected_rank", metadata.get("n_components"))
    existing_rank = None
    if existing_rank_raw is not None:
        try:
            existing_rank = int(existing_rank_raw)
        except (TypeError, ValueError):
            existing_rank = None

    if existing_rank == selected_rank:
        log(
            "Enterosignature-selected rank matches the existing NMF baseline; "
            "skipping recomputation."
        )
        return results

    log(
        "Refreshing NMF baseline metrics using the enterosignature-selected "
        f"rank (k={selected_rank})."
    )

    counts = load_counts(matrix_path, log1p=False)
    n_splits = len(nmf_result.fold_metrics) if nmf_result.fold_metrics else 5
    train_fraction = metadata.get("train_fraction", 0.9)
    try:
        train_fraction = float(train_fraction)
    except (TypeError, ValueError):
        train_fraction = 0.9
    log1p = bool(metadata.get("log1p", True))
    nmf_kwargs_raw = metadata.get("nmf_kwargs")
    nmf_kwargs = dict(nmf_kwargs_raw) if isinstance(nmf_kwargs_raw, Mapping) else None
    resolved_seeds = [int(s) for s in seeds]
    if not resolved_seeds:
        raise SystemExit(
            "At least one seed must be provided for the NMF baseline refresh."
        )
    log(
        "Refreshing NMF baseline across seeds "
        + ", ".join(str(s) for s in resolved_seeds)
    )

    taxonomy_eval = None
    taxonomy_levels = metadata.get("taxonomy_levels")
    if isinstance(taxonomy_levels, list) and taxonomy_levels:
        try:
            tax_struct = build_taxonomy_structures(
                matrix_path,
                taxonomy_path=taxonomy_path,
                levels=[str(level) for level in taxonomy_levels],
                lap_w=[],
                verbose=verbose,
            )
        except Exception as exc:
            raise SystemExit(
                f"Failed to build taxonomy evaluation matrices: {exc}"
            ) from exc
        taxonomy_eval = {
            level: mat for level, mat in tax_struct.get("A_mats", {}).items()
        }
    else:
        has_tax_metrics = any(
            key.startswith(("mae_tax_", "rmse_tax_", "r2_tax_"))
            for key in nmf_result.mean_metrics
        )
        if has_tax_metrics:
            log(
                "Existing NMF baseline includes taxonomy-aware metrics, but no "
                "taxonomy levels were recorded; those metrics will be omitted "
                "from the refreshed baseline."
            )

    try:
        refreshed = cross_validate_nmf_multi_seed(
            counts,
            n_components=selected_rank,
            n_splits=n_splits,
            train_fraction=train_fraction,
            log1p=log1p,
            nmf_kwargs=nmf_kwargs,
            seeds=resolved_seeds,
            taxonomy_eval=taxonomy_eval,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to refresh NMF baseline metrics: {exc}") from exc

    refreshed_metadata = dict(refreshed.metadata or {})
    refreshed_metadata["selected_rank"] = selected_rank
    refreshed_metadata["rank_source"] = "enterosignatures"
    refreshed_metadata["seeds"] = resolved_seeds
    refreshed_result = CrossValResult(
        refreshed.fold_metrics,
        refreshed.mean_metrics,
        refreshed.std_metrics,
        refreshed_metadata,
    )

    updated = dict(results)
    updated["nmf"] = refreshed_result
    return updated


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    verbose = bool(args.verbose)

    def _log(message: str) -> None:
        if verbose:
            print(message)

    def _render_progress(current: int, total: int, message: str) -> None:
        if total <= 0:
            total = 1
        bar_width = 28
        filled = int(bar_width * current / total)
        bar = "=" * filled + "-" * (bar_width - filled)
        line = f"\r[{bar}] {current}/{total}"
        if message:
            line = f"{line} {message}"
        sys.stdout.write(line)
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    _log("Verbose logging enabled (use --no-verbose to silence).")
    _log(f"Inputs: {', '.join(args.input)}")
    _log(f"Counts matrix: {args.matrix}")
    _log(f"Taxonomy table: {args.taxonomy}")
    _log(
        "Output specs: "
        f"metrics={args.output or 'interactive'}, "
        f"ordinations={args.ordinations_output or 'auto'}, "
        f"enterosignatures={args.enterosignature_output or 'auto'}, "
        f"rank-selection={args.rank_selection_output or 'auto'}, "
        f"comparison={args.comparison_output or 'auto'}, "
        f"agreement={args.agreement_output or 'auto'}, "
        f"signature-genus={args.signature_genus_output or 'none'}"
    )
    _log(
        "Optional outputs: "
        f"geometry={args.geometry_output or 'none'}, "
        f"geometry-plot={args.geometry_plot_output or 'none'}, "
        f"procrustes={args.procrustes_output or 'none'}, "
        f"procrustes-stats={args.procrustes_stats_output or 'none'}, "
        f"clustering={args.clustering_output or 'none'}, "
        f"contingency={args.contingency_output or 'none'}, "
        f"contingency-plot={args.contingency_plot_output or 'none'}"
    )
    _log(
        "Model selection settings: "
        f"k-range={args.k_range}, alpha-range={args.alpha_range}, "
        f"bicross-folds={args.bicross_folds}, "
        f"bicross-repetitions={args.bicross_repetitions}, "
        f"nmf-max-iter={args.nmf_max_iter}"
    )
    _log(
        "Clustering settings: "
        f"random-state={args.random_state}, max-iter={args.max_iter}, "
        f"clusters={args.clusters or 'auto'}, signatures={args.signatures or 'auto'}"
    )
    _log(
        "Additional options: "
        f"matrix-log1p={args.matrix_log1p}, "
        f"mantel-permutations={args.mantel_permutations}, "
        f"include-original-distance={args.include_original_distance}, "
        f"signature-genus-top={args.signature_genus_top}, "
        f"reuse-nmf-results={args.reuse_nmf_results}"
    )

    _log(f"Loading benchmark results from {len(args.input)} input(s).")
    results = _load_results(args.input)
    _log(f"Loaded {len(results)} method result(s).")
    renames = _parse_renames(args.rename)
    if renames:
        _log(f"Applying {len(renames)} rename(s) to method labels.")
    embeddings = _parse_embedding_specs(args.embedding, args.latent)
    if embeddings:
        _log(f"Loaded {len(embeddings)} embedding specification(s).")
        _log("Embedding labels: " + ", ".join(sorted(embeddings)))

    if args.signatures is None:
        k_min, k_max = _parse_range(args.k_range, name="k-range")
        k_values = list(range(k_min, k_max + 1))
    else:
        k_min = k_max = args.signatures
        k_values = [args.signatures]
        _log(f"Using requested enterosignatures: k={args.signatures}.")

    alpha_min, alpha_max = _parse_range(args.alpha_range, name="alpha-range")
    alpha_values = list(range(alpha_min, alpha_max + 1))

    if args.signatures is None:
        _log(
            "Computing enterosignatures with "
            f"k={k_min}-{k_max}, clusters={args.clusters or 'auto'}, "
            f"alpha={alpha_min}-{alpha_max}, bicross_folds={args.bicross_folds}, "
            f"bicross_repetitions={args.bicross_repetitions}."
        )
    else:
        _log(
            "Computing enterosignatures with "
            f"k={args.signatures}, clusters={args.clusters or 'auto'}, "
            f"alpha={alpha_min}-{alpha_max}, bicross_folds={args.bicross_folds}, "
            f"bicross_repetitions={args.bicross_repetitions}."
        )
    enterosig = compute_enterosignatures(
        args.matrix,
        args.taxonomy,
        args.signatures,
        args.clusters,
        random_state=args.random_state,
        k_range=k_values,
        alpha_range=alpha_values,
        max_iter=args.nmf_max_iter,
        bicross_folds=args.bicross_folds,
        bicross_repetitions=args.bicross_repetitions,
        cluster_max_iter=args.max_iter,
        progress_callback=_render_progress,
    )
    if enterosig.median_cosine_similarity:
        print(
            "Median cosine similarity by k: "
            + ", ".join(
                f"{k}={v:.4f}" for k, v in enterosig.median_cosine_similarity.items()
            )
        )
    if enterosig.median_r2_score:
        print(
            "Median R^2 by k: "
            + ", ".join(f"{k}={v:.4f}" for k, v in enterosig.median_r2_score.items())
        )
    if enterosig.cosine_similarity_scores:
        rank_output = args.rank_selection_output
        if rank_output is None:
            rank_output = "enterosignatures_rank_selection.pdf"
            _log("Rank selection output not provided; using default.")
        rank_path = Path(rank_output)
        _log(f"Saving rank selection plot to {rank_path}.")
        _plot_rank_selection(
            cosine_scores=enterosig.cosine_similarity_scores,
            r2_scores=enterosig.r2_scores,
            selected_k=enterosig.n_components,
            title="Enterosignature rank selection",
            output=rank_path,
        )
        print(f"Saved enterosignature rank selection plot to {rank_path.resolve()}")
    if enterosig.mean_explained_variance:
        print(
            "Mean R^2 by alpha: "
            + ", ".join(
                f"{k}={v:.4f}" for k, v in enterosig.mean_explained_variance.items()
            )
        )
    print(
        "Selected enterosignatures: "
        f"k={enterosig.n_components}, clusters={enterosig.n_clusters}, "
        f"alpha={enterosig.alpha:.3f}"
    )

    nmf_refresh_seeds = (
        [args.seed] if args.seed is not None else list(args.seeds)
    )
    _log(f"NMF reconstruction refresh seeds: {nmf_refresh_seeds}")
    results = _refresh_nmf_baseline(
        results,
        selected_rank=enterosig.n_components,
        matrix_path=args.matrix,
        taxonomy_path=args.taxonomy,
        reuse=args.reuse_nmf_results,
        verbose=verbose,
        log=_log,
        seeds=nmf_refresh_seeds,
    )

    ordinations = _prepare_ordinations(args.matrix, args.matrix_log1p, embeddings, results)
    if ordinations:
        _log(f"Prepared ordinations for {len(ordinations)} dataset(s).")

    renamed_results: Dict[str, CrossValResult] = {}
    for name, result in results.items():
        new_name = renames.get(name, name)
        if new_name in renamed_results:
            raise SystemExit(
                f"Renaming results would create duplicate label '{new_name}'."
            )
        renamed_results[new_name] = result

    baseline = renames.get(args.baseline, args.baseline)
    if baseline and baseline not in renamed_results:
        raise SystemExit(
            f"Baseline '{baseline}' is not present in the loaded results after applying renames."
        )
    latent_dims = _collect_latent_dims(renamed_results)

    all_metric_keys = None
    for result in renamed_results.values():
        keys = set(result.mean_metrics.keys())
        all_metric_keys = keys if all_metric_keys is None else all_metric_keys & keys
    if not all_metric_keys:
        raise SystemExit("No common metrics found across the provided results.")

    requested_metrics = []
    if args.metrics:
        for metric in args.metrics:
            if metric not in all_metric_keys:
                raise SystemExit(
                    f"Metric '{metric}' is not present in all results; available: {sorted(all_metric_keys)}"
                )
            if metric not in requested_metrics:
                requested_metrics.append(metric)
    else:
        requested_metrics = sorted(all_metric_keys)

    if not requested_metrics:
        raise SystemExit("No metrics selected for plotting.")

    stat_metrics = [metric for metric in ("mae", "rmse") if metric in all_metric_keys]
    if stat_metrics:
        _print_pairwise_stats(renamed_results, stat_metrics)

    _log(f"Rendering figures for metrics: {', '.join(requested_metrics)}.")
    figsize = _parse_figsize(args.figsize)
    output_spec = args.output
    for metric in requested_metrics:
        figure_output: Path | None = None
        if output_spec:
            base = Path(output_spec)
            if len(requested_metrics) == 1:
                figure_output = base
            else:
                if base.suffix:
                    figure_output = base.with_name(f"{base.stem}_{metric}{base.suffix}")
                else:
                    figure_output = base / f"{metric}.pdf"
            figure_output.parent.mkdir(parents=True, exist_ok=True)

        if figure_output:
            _log(f"Generating '{metric}' figure -> {figure_output}.")
        else:
            _log(f"Generating '{metric}' figure (no output path specified).")
        figure, _axes = plot_benchmark_figure(
            renamed_results,
            metric=metric,
            title=args.title,
            baseline=baseline,
            figsize=figsize,
            output=str(figure_output) if figure_output else None,
        )
        _close_figure(figure)

        if figure_output:
            print(f"Saved figure to {figure_output.resolve()}")
        else:
            print(f"Generated {metric} figure; use --output to save it.")

    if ordinations:
        _log("Rendering ordination grid for PCA/t-SNE.")
        ordination_order = _order_methods_by_metric(renamed_results, requested_metrics[0])
        ord_output_spec = args.ordinations_output
        ord_output: Path | None = None
        if ord_output_spec:
            ord_output = Path(ord_output_spec)
        elif output_spec:
            base = Path(output_spec)
            if base.suffix:
                ord_output = base.with_name(f"{base.stem}_ordinations{base.suffix}")
            else:
                ord_output = base / "ordinations.pdf"
        if ord_output:
            ord_output.parent.mkdir(parents=True, exist_ok=True)

        ord_title = args.title or "PCA/t-SNE ordinations"
        ord_figure, _ord_axes = plot_ordination_grid(
            ordinations,
            title=ord_title,
            output=str(ord_output) if ord_output else None,
            order=ordination_order,
            latent_dims=latent_dims,
        )
        _close_figure(ord_figure)

        if ord_output:
            print(f"Saved ordination figure to {ord_output.resolve()}")
        else:
            print("Generated ordination figure; use --ordinations-output to save it.")

    if args.signature_genus_top < 1:
        raise SystemExit("--signature-genus-top must be at least 1.")
    genus_lines = _format_signature_genus_summary(
        basis=enterosig.basis,
        genus_names=enterosig.genus_names,
        top_n=args.signature_genus_top,
    )
    print("Top genera per enterosignature:")
    for line in genus_lines:
        print(line)

    if args.signature_genus_output:
        output_path = Path(args.signature_genus_output)
        _log(f"Saving enterosignature genus summary to {output_path}.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("signature\trank\tgenus\tweight\n")
            for signature_idx in range(enterosig.basis.shape[0]):
                weights = enterosig.basis[signature_idx]
                top_indices = list(reversed(np.argsort(weights)))[: args.signature_genus_top]
                for rank, genus_idx in enumerate(top_indices, start=1):
                    handle.write(
                        f"{signature_idx + 1}\t{rank}\t{enterosig.genus_names[genus_idx]}\t"
                        f"{weights[genus_idx]:.6f}\n"
                    )
        print(f"Saved enterosignature genus summary to {output_path.resolve()}")

    if ordinations:
        _log("Rendering enterosignature ordination grid.")
        enterosig_output = Path(args.enterosignature_output or "enterosignatures_ordinations.pdf")
        _log(f"Saving enterosignature ordination grid to {enterosig_output}.")
        enterosig_output.parent.mkdir(parents=True, exist_ok=True)
        enterosig_fig, _ = plot_enterosignature_ordination_grid(
            ordinations,
            enterosig.labels,
            signature_weights=enterosig.weights,
            title=args.enterosignature_title,
            output=str(enterosig_output),
            order=ordination_order if ordinations else None,
            latent_dims=latent_dims,
        )
        _close_figure(enterosig_fig)
        print(f"Saved enterosignature ordination figure to {enterosig_output.resolve()}")

    mixture = enterosig.weights
    mixture_row_sums = mixture.sum(axis=1, keepdims=True)
    mixture_row_sums[mixture_row_sums == 0] = 1.0
    mixture_norm = mixture / mixture_row_sums

    if embeddings:
        _log("Computing enterosignature agreement for embeddings.")
        agreement: Dict[str, float] = {}
        embedding_dims: Dict[str, int] = {}
        compositions: Dict[str, np.ndarray] = {}
        for label, path in embeddings.items():
            try:
                matrix, sample_names = load_latent_embeddings(
                    path, sample_names=enterosig.sample_names
                )
            except Exception as exc:
                raise SystemExit(
                    f"Failed to load embeddings '{label}': {exc}"
                ) from exc
            if sample_names != enterosig.sample_names:
                raise SystemExit(
                    f"Embedding '{label}' samples do not match the counts matrix ordering."
                )
            embedding_dims[label] = int(matrix.shape[1])
            try:
                assigned = cluster_embeddings(
                    matrix,
                    enterosig.n_clusters,
                    random_state=args.random_state,
                    max_iter=args.max_iter,
                )
            except Exception as exc:
                raise SystemExit(
                    f"Failed to cluster embeddings for '{label}': {exc}"
                ) from exc
            agreement[label] = float(
                adjusted_rand_score(enterosig.labels, assigned)
            )
            cluster_composition = np.zeros(
                (enterosig.n_clusters, enterosig.n_components), dtype=float
            )
            for cluster_idx in range(enterosig.n_clusters):
                members = mixture_norm[assigned == cluster_idx]
                if members.size == 0:
                    continue
                cluster_composition[cluster_idx] = members.mean(axis=0)
            compositions[label] = cluster_composition

        print("Enterosignature agreement by embedding (adjusted Rand index):")
        for label in sorted(agreement, key=agreement.get, reverse=True):
            dims = embedding_dims.get(label)
            dims_text = f"{dims}D" if dims is not None else "unknown dims"
            print(f"  {label}: {agreement[label]:.4f} ({dims_text})")

        agreement_output = Path(
            args.agreement_output or "enterosignatures_agreement_ari.pdf"
        )
        _log(f"Saving enterosignature agreement plot to {agreement_output}.")
        agreement_output.parent.mkdir(parents=True, exist_ok=True)
        agreement_fig, _ = plot_enterosignature_agreement(
            agreement,
            title=args.agreement_title,
            output=str(agreement_output),
        )
        _close_figure(agreement_fig)
        print(
            f"Saved enterosignature agreement plot to {agreement_output.resolve()}"
        )

        comparison_output = Path(
            args.comparison_output or "enterosignatures_agreement.pdf"
        )
        _log(f"Saving enterosignature composition plot to {comparison_output}.")
        comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison_fig, _ = plot_enterosignature_comparison(
            compositions,
            title=args.comparison_title,
            output=str(comparison_output),
            enterosignature_labels=[
                f"Signature {idx + 1}" for idx in range(enterosig.n_components)
            ],
        )
        _close_figure(comparison_fig)
        print(
            f"Saved enterosignature composition plot to {comparison_output.resolve()}"
        )
    else:
        print("No embeddings provided; skipping enterosignature composition plot.")
        return

    distance_matrices: Dict[str, np.ndarray] = {
        "enterosignature": _compute_distance_matrix(mixture_norm, metric="euclidean")
    }
    if args.include_original_distance:
        distance_matrices["genus_abundance"] = _compute_distance_matrix(
            enterosig.genus_abundance, metric="braycurtis"
        )

    embedding_matrices: Dict[str, np.ndarray] = {}
    for label, path in embeddings.items():
        latent, latent_names = load_latent_embeddings(
            path, sample_names=enterosig.sample_names
        )
        if latent_names != enterosig.sample_names:
            raise SystemExit(
                f"Embedding '{label}' samples do not match the counts matrix ordering."
            )
        embedding_matrices[label] = latent
        distance_matrices[label] = _compute_distance_matrix(latent, metric="euclidean")

    mantel_rows: list[list[str]] = []
    for label in embeddings.keys():
        es_corr, es_p = _mantel_test(
            distance_matrices["enterosignature"],
            distance_matrices[label],
            permutations=args.mantel_permutations,
            random_state=args.random_state,
        )
        mantel_rows.append(
            [
                "enterosignature",
                label,
                f"{es_corr:.4f}",
                f"{es_p:.6f}",
                str(args.mantel_permutations),
            ]
        )
        if "genus_abundance" in distance_matrices:
            gx_corr, gx_p = _mantel_test(
                distance_matrices["genus_abundance"],
                distance_matrices[label],
                permutations=args.mantel_permutations,
                random_state=args.random_state,
            )
            mantel_rows.append(
                [
                    "genus_abundance",
                    label,
                    f"{gx_corr:.4f}",
                    f"{gx_p:.6f}",
                    str(args.mantel_permutations),
                ]
            )

    if args.geometry_output:
        output_path = Path(args.geometry_output)
        _write_tsv(
            output_path,
            ["space_a", "space_b", "correlation", "p_value", "permutations"],
            mantel_rows,
        )
        print(f"Saved distance geometry summary to {output_path.resolve()}")

    if args.geometry_plot_output:
        output_path = Path(args.geometry_plot_output)
        for label in embeddings.keys():
            es_dist = _extract_upper_triangle(distance_matrices["enterosignature"])
            emb_dist = _extract_upper_triangle(distance_matrices[label])
            plot_path = _format_path_suffix(output_path, f"mantel_enterosignature_{label}")
            _plot_distance_scatter(
                distances_a=es_dist,
                distances_b=emb_dist,
                labels=("Enterosignature", label),
                title=f"Distance comparison: Enterosignature vs {label}",
                output=plot_path,
            )
            print(f"Saved distance scatter plot to {plot_path.resolve()}")
            if "genus_abundance" in distance_matrices:
                gx_dist = _extract_upper_triangle(distance_matrices["genus_abundance"])
                gx_plot_path = _format_path_suffix(
                    output_path, f"mantel_genus_abundance_{label}"
                )
                _plot_distance_scatter(
                    distances_a=gx_dist,
                    distances_b=emb_dist,
                    labels=("Genus abundance", label),
                    title=f"Distance comparison: Genus abundance vs {label}",
                    output=gx_plot_path,
                )
                print(f"Saved distance scatter plot to {gx_plot_path.resolve()}")

    procrustes_rows: list[list[str]] = []
    if args.procrustes_output or args.procrustes_stats_output:
        es_mds = _compute_mds(
            distance_matrices["enterosignature"],
            random_state=args.random_state,
        )
        for label in embeddings.keys():
            vae_mds = _compute_mds(
                distance_matrices[label],
                random_state=args.random_state,
            )
            es_proc, vae_proc, disparity = procrustes(es_mds, vae_mds)
            procrustes_rows.append([label, f"{disparity:.6f}"])
            if args.procrustes_output:
                output_path = _format_path_suffix(
                    Path(args.procrustes_output), f"procrustes_{label}"
                )
                _plot_procrustes(
                    coords_a=es_proc,
                    coords_b=vae_proc,
                    labels=["Enterosignature", label],
                    title=f"Procrustes alignment: Enterosignature vs {label}",
                    output=output_path,
                )
                print(f"Saved Procrustes alignment plot to {output_path.resolve()}")

    if args.procrustes_stats_output and procrustes_rows:
        output_path = Path(args.procrustes_stats_output)
        _write_tsv(output_path, ["embedding", "disparity"], procrustes_rows)
        print(f"Saved Procrustes summary to {output_path.resolve()}")

    cluster_rows: list[list[str]] = []
    contingency_rows: list[list[str]] = []
    es_labels = enterosig.labels
    es_distance = distance_matrices["enterosignature"]
    es_unique = np.unique(es_labels)
    es_silhouette = None
    if len(es_unique) > 1 and es_distance.shape[0] > len(es_unique):
        es_silhouette = float(
            silhouette_score(es_distance, es_labels, metric="precomputed")
        )
    for label, latent in embedding_matrices.items():
        vae_labels = cluster_embeddings(
            latent,
            enterosig.n_clusters,
            random_state=args.random_state,
            max_iter=args.max_iter,
        )
        ari = adjusted_rand_score(es_labels, vae_labels)
        ami = adjusted_mutual_info_score(es_labels, vae_labels)
        vae_distance = distance_matrices[label]
        vae_unique = np.unique(vae_labels)
        vae_silhouette = None
        if len(vae_unique) > 1 and vae_distance.shape[0] > len(vae_unique):
            vae_silhouette = float(
                silhouette_score(vae_distance, vae_labels, metric="precomputed")
            )
        cluster_rows.append(
            [
                label,
                f"{ari:.4f}",
                f"{ami:.4f}",
                f"{es_silhouette:.4f}" if es_silhouette is not None else "NA",
                f"{vae_silhouette:.4f}" if vae_silhouette is not None else "NA",
            ]
        )

        if args.contingency_output:
            labels_sorted = sorted(set(es_labels.tolist()))
            vae_sorted = sorted(set(vae_labels.tolist()))
            contingency_rows.append([f"{label} contingency"])
            contingency_rows.append(["es_label", *[str(v) for v in vae_sorted]])
            for es_label in labels_sorted:
                row = [str(es_label)]
                for vae_label in vae_sorted:
                    count = int(np.sum((es_labels == es_label) & (vae_labels == vae_label)))
                    row.append(str(count))
                contingency_rows.append(row)
            contingency_rows.append([])

        if args.contingency_plot_output:
            labels_sorted = sorted(set(es_labels.tolist()))
            vae_sorted = sorted(set(vae_labels.tolist()))
            table = np.zeros((len(labels_sorted), len(vae_sorted)), dtype=int)
            for es_idx, es_label in enumerate(labels_sorted):
                for vae_idx, vae_label in enumerate(vae_sorted):
                    table[es_idx, vae_idx] = int(
                        np.sum((es_labels == es_label) & (vae_labels == vae_label))
                    )
            plot_path = _format_path_suffix(
                Path(args.contingency_plot_output), f"contingency_{label}"
            )
            _plot_contingency_heatmap(
                table=table,
                row_labels=[str(val) for val in labels_sorted],
                col_labels=[str(val) for val in vae_sorted],
                title=f"Clustering contingency: Enterosignature vs {label}",
                output=plot_path,
            )
            print(f"Saved contingency heatmap to {plot_path.resolve()}")

    if args.clustering_output and cluster_rows:
        output_path = Path(args.clustering_output)
        _write_tsv(
            output_path,
            [
                "embedding",
                "ari",
                "ami",
                "silhouette_enterosignature",
                "silhouette_embedding",
            ],
            cluster_rows,
        )
        print(f"Saved clustering agreement summary to {output_path.resolve()}")

    if args.contingency_output and contingency_rows:
        output_path = Path(args.contingency_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in contingency_rows:
                handle.write("\t".join(row) + "\n")
        print(f"Saved clustering contingency table to {output_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
