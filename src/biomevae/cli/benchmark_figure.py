"""CLI entry point to render benchmark reconstruction figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from biomevae.data import load_matrix
from biomevae.enterosignatures import load_latent_embeddings
from biomevae.reconstruction import (
    CrossValResult,
    compute_ordinations,
    compute_pairwise_metric_stats,
    adjust_pvalues_bh,
    adjust_pvalues_bonferroni,
    fit_nmf_embeddings,
    plot_benchmark_figure,
    plot_ordination_grid,
)

from ._recon_cli import dict_to_result, load_json


def _parse_figsize(value: str) -> tuple[float, float]:
    try:
        width_str, height_str = value.lower().split("x", 1)
        return float(width_str), float(height_str)
    except (ValueError, TypeError) as exc:
        raise SystemExit(
            "--figsize must be provided as WIDTHxHEIGHT (e.g. 7x4)."
        ) from exc


def _parse_renames(items: list[str] | None) -> Dict[str, str]:
    renames: Dict[str, str] = {}
    if not items:
        return renames
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid rename '{item}'. Use old=new format.")
        old, new = (part.strip() for part in item.split("=", 1))
        if not old or not new:
            raise SystemExit("Rename assignments must include both old and new labels.")
        renames[old] = new
    return renames


def _parse_embedding_specs(
    specs: list[str] | None, legacy_latent: str | None
) -> Dict[str, str]:
    embeddings: Dict[str, str] = {}
    if legacy_latent:
        default_label = Path(legacy_latent).stem or "latent"
        embeddings[default_label] = legacy_latent
    if not specs:
        return embeddings
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"Invalid embedding specification '{spec}'. Use NAME=PATH format."
            )
        name, path = (part.strip() for part in spec.split("=", 1))
        if not name:
            raise SystemExit("Embedding labels cannot be empty.")
        if not path:
            raise SystemExit("Embedding paths cannot be empty.")
        if name in embeddings:
            raise SystemExit(f"Embedding label '{name}' provided more than once.")
        embeddings[name] = path
    return embeddings


def _load_results(paths: list[str]) -> Dict[str, CrossValResult]:
    results: Dict[str, CrossValResult] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise SystemExit(
                f"Top-level structure in '{path}' must be a JSON object mapping method names to results."
            )
        for name, raw in payload.items():
            if not isinstance(raw, Mapping):
                raise SystemExit(
                    f"Entry '{name}' in '{path}' must be an object describing the metrics."
                )
            if name in results:
                raise SystemExit(
                    f"Method '{name}' appears multiple times across inputs; rename duplicates first."
                )
            results[name] = dict_to_result(raw)
    if not results:
        raise SystemExit("No methods found in the provided inputs.")
    return results


def _order_methods_by_metric(
    results: Mapping[str, CrossValResult], metric: str
) -> list[str]:
    ordered: list[tuple[str, float]] = []
    for name, result in results.items():
        if metric not in result.mean_metrics:
            raise SystemExit(
                f"Metric '{metric}' is not present in all results; available: {sorted(result.mean_metrics.keys())}"
            )
        ordered.append((name, float(result.mean_metrics[metric])))
    ordered.sort(key=lambda item: item[1])
    return [name for name, _ in ordered]


def _print_pairwise_stats(
    results: Mapping[str, CrossValResult],
    metrics: list[str],
) -> None:
    if len(results) < 2:
        print("Pairwise statistical comparisons require at least two methods.")
        return
    for metric in metrics:
        comparisons = compute_pairwise_metric_stats(results, metric)
        if not comparisons:
            continue
        raw_pvalues = [float(row["p_value"]) for row in comparisons]
        pvals_bh = adjust_pvalues_bh(raw_pvalues)
        pvals_bonferroni = adjust_pvalues_bonferroni(raw_pvalues)
        for index, row in enumerate(comparisons):
            row["p_value_bh"] = pvals_bh[index]
            row["p_value_bonferroni"] = pvals_bonferroni[index]
        print(f"Pairwise sign-test comparisons for '{metric}':")
        header = (
            "  model_a vs model_b | mean_diff | median_diff | n | "
            "p_value | p_bh | p_bonf"
        )
        print(header)
        for row in comparisons:
            print(
                "  "
                f"{row['model_a']} vs {row['model_b']} | "
                f"{row['mean_diff']:.4g} | {row['median_diff']:.4g} | "
                f"{row['n']} | {row['p_value']:.3g} | "
                f"{row['p_value_bh']:.3g} | {row['p_value_bonferroni']:.3g}"
            )
        print("")


def _collect_latent_dims(
    results: Mapping[str, CrossValResult],
) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for name, result in results.items():
        metadata = result.metadata or {}
        raw_dim = metadata.get("latent_dim")
        if raw_dim is None:
            raw_dim = metadata.get("selected_rank", metadata.get("n_components"))
        if raw_dim is None:
            continue
        try:
            dims[name] = int(raw_dim)
        except (TypeError, ValueError):
            continue
    return dims


def _prepare_ordinations(
    matrix_path: str | None,
    log1p: bool,
    embeddings: Dict[str, str],
    results: Mapping[str, CrossValResult] | None = None,
) -> Dict[str, Dict[str, np.ndarray]] | None:
    if not matrix_path:
        if embeddings:
            raise SystemExit("--matrix is required when specifying embeddings")
        return None

    try:
        counts, sample_names = load_matrix(matrix_path, log1p=False)
    except SystemExit as exc:
        raise SystemExit(str(exc))

    ordinations: Dict[str, Dict[str, np.ndarray]] = {}
    try:
        base_ords = compute_ordinations(counts, log1p=log1p)
    except Exception as exc:
        raise SystemExit(f"Failed to compute PCA/t-SNE ordinations for the counts matrix: {exc}") from exc

    def _extract(space: Mapping[str, np.ndarray], key: str) -> np.ndarray:
        if key not in space:
            raise KeyError(key)
        return np.asarray(space[key])

    try:
        counts_ords: Dict[str, np.ndarray] = {
            "pca": _extract(base_ords, "pca"),
            "tsne": _extract(base_ords, "tsne"),
        }
        if "umap" in base_ords:
            counts_ords["umap"] = _extract(base_ords, "umap")
        ordinations["counts"] = counts_ords
    except KeyError as exc:
        raise SystemExit(
            f"Ordination result '{exc.args[0]}' missing for counts matrix computation"
        ) from exc

    if results and "nmf" in results:
        metadata = results["nmf"].metadata or {}
        n_components_raw = metadata.get("selected_rank", metadata.get("n_components"))
        try:
            n_components = int(n_components_raw)
        except (TypeError, ValueError):
            n_components = 0
        if n_components > 0:
            nmf_log1p = bool(metadata.get("log1p", True))
            nmf_kwargs_raw = metadata.get("nmf_kwargs")
            nmf_kwargs = dict(nmf_kwargs_raw) if isinstance(nmf_kwargs_raw, Mapping) else None
            random_state_raw = metadata.get("random_state")
            random_state = None
            if random_state_raw is not None:
                try:
                    random_state = int(random_state_raw)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        "Metadata for the NMF baseline contains an invalid random_state value"
                    ) from exc
            try:
                nmf_latent = fit_nmf_embeddings(
                    counts,
                    n_components=n_components,
                    log1p=nmf_log1p,
                    nmf_kwargs=nmf_kwargs,
                    random_state=random_state,
                )
            except Exception as exc:
                raise SystemExit(f"Failed to fit NMF embeddings for ordinations: {exc}") from exc
            try:
                nmf_ords = compute_ordinations(counts, log1p=log1p, latent=nmf_latent)
            except Exception as exc:
                raise SystemExit(
                    f"Failed to compute PCA/t-SNE ordinations for the NMF embedding: {exc}"
                ) from exc
            pca_key = "pca_latent" if "pca_latent" in nmf_ords else "pca"
            tsne_key = "tsne_latent" if "tsne_latent" in nmf_ords else "tsne"
            umap_key = "umap_latent" if "umap_latent" in nmf_ords else "umap"
            try:
                nmf_ord_dict: Dict[str, np.ndarray] = {
                    "pca": _extract(nmf_ords, pca_key),
                    "tsne": _extract(nmf_ords, tsne_key),
                }
                if umap_key in nmf_ords:
                    nmf_ord_dict["umap"] = _extract(nmf_ords, umap_key)
                ordinations["nmf"] = nmf_ord_dict
            except KeyError as exc:
                raise SystemExit(
                    f"Ordination result '{exc.args[0]}' missing for the NMF embedding"
                ) from exc

    for label, path in embeddings.items():
        try:
            latent, latent_names = load_latent_embeddings(
                path, sample_names=sample_names
            )
        except Exception as exc:
            raise SystemExit(f"Failed to load latent embeddings '{path}': {exc}") from exc
        if latent_names != sample_names:
            raise SystemExit(
                f"Embedding '{label}' samples do not match the counts matrix ordering."
            )
        try:
            latent_ords = compute_ordinations(counts, log1p=log1p, latent=latent)
        except Exception as exc:
            raise SystemExit(
                f"Failed to compute PCA/t-SNE ordinations for embedding '{label}': {exc}"
            ) from exc

        pca_key = "pca_latent" if "pca_latent" in latent_ords else "pca"
        tsne_key = "tsne_latent" if "tsne_latent" in latent_ords else "tsne"
        umap_key = "umap_latent" if "umap_latent" in latent_ords else "umap"
        try:
            emb_ord_dict: Dict[str, np.ndarray] = {
                "pca": _extract(latent_ords, pca_key),
                "tsne": _extract(latent_ords, tsne_key),
            }
            if umap_key in latent_ords:
                emb_ord_dict["umap"] = _extract(latent_ords, umap_key)
            ordinations[label] = emb_ord_dict
        except KeyError as exc:
            raise SystemExit(
                f"Ordination result '{exc.args[0]}' missing for embedding '{label}'"
            ) from exc

    return ordinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("biomevae-benchmark-figure")
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
        help="Optional path where the figure should be written",
    )
    parser.add_argument(
        "--rename",
        nargs="*",
        metavar="OLD=NEW",
        help="Rename method labels before plotting (repeatable)",
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Optional counts matrix used to compute PCA/t-SNE overlays",
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    verbose = bool(args.verbose)

    def _log(message: str) -> None:
        if verbose:
            print(message)

    _log(f"Loading benchmark results from {len(args.input)} input(s).")
    results = _load_results(args.input)
    _log(f"Loaded {len(results)} method result(s).")
    renames = _parse_renames(args.rename)
    if renames:
        _log(f"Applying {len(renames)} rename(s) to method labels.")
    embeddings = _parse_embedding_specs(args.embedding, args.latent)
    if embeddings:
        _log(f"Loaded {len(embeddings)} embedding specification(s).")
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

        try:  # Ensure resources are released when running headless
            import matplotlib.pyplot as plt

            plt.close(figure)
        except ImportError:  # pragma: no cover - matplotlib already validated upstream
            pass

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

        try:
            import matplotlib.pyplot as plt

            plt.close(ord_figure)
        except ImportError:  # pragma: no cover
            pass

        if ord_output:
            print(f"Saved ordination figure to {ord_output.resolve()}")
        else:
            print("Generated ordination figure; use --ordinations-output to save it.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
