"""CLI entry point to build LaTeX slides for benchmark figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from biomevae.reconstruction import (
    CrossValResult,
    plot_benchmark_figure,
    plot_ordination_grid,
)

from .benchmark_figure import (
    _collect_latent_dims,
    _load_results,
    _order_methods_by_metric,
    _parse_figsize,
    _parse_embedding_specs,
    _parse_renames,
    _print_pairwise_stats,
    _prepare_ordinations,
)


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in free-form text."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = []
    for char in text:
        escaped.append(replacements.get(char, char))
    return "".join(escaped)


def _escape_path(path: str) -> str:
    """Escape LaTeX special characters within file paths."""

    # ``\detokenize`` keeps the exact path even if it contains special characters.
    return rf"\detokenize{{{path}}}"


def _rename_results(
    results: Dict[str, CrossValResult], renames: Dict[str, str]
) -> Dict[str, CrossValResult]:
    renamed: Dict[str, CrossValResult] = {}
    for name, result in results.items():
        new_name = renames.get(name, name)
        if new_name in renamed:
            raise SystemExit(
                f"Renaming results would create duplicate label '{new_name}'."
            )
        renamed[new_name] = result
    return renamed


def _collect_metric_rows(
    results: Dict[str, CrossValResult], metric: str
) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for name, result in results.items():
        try:
            mean_value = float(result.mean_metrics[metric])
            std_value = float(result.std_metrics[metric])
        except KeyError as exc:
            raise SystemExit(
                f"Metric '{metric}' is missing for method '{name}'."
            ) from exc
        rows.append((name, mean_value, std_value))
    rows.sort(key=lambda row: row[1])
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("biomevae-benchmark-slides")
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
            "Metric key used both for plotting and the summary table (repeatable). "
            "When omitted, all metrics shared across the methods are rendered."
        ),
    )
    parser.add_argument("--title", default="Benchmark Results", help="Slide deck title")
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Optional subtitle displayed on the title slide",
    )
    parser.add_argument(
        "--author", default=None, help="Author shown on the title slide"
    )
    parser.add_argument(
        "--date", default="\\today", help="Date string for the title slide"
    )
    parser.add_argument(
        "--theme",
        default="Madrid",
        help="Beamer theme to use for the generated slides",
    )
    parser.add_argument(
        "--baseline",
        default="nmf",
        help="Method name highlighted as the baseline in the table",
    )
    parser.add_argument(
        "--figsize",
        default="7x4",
        help="Figure size specified as WIDTHxHEIGHT in inches (default: 7x4)",
    )
    parser.add_argument(
        "--figure-output",
        default="benchmark_figure.pdf",
        help="Path where the benchmark figure will be written",
    )
    parser.add_argument(
        "--slides-output",
        default="benchmark_slides.tex",
        help="Path where the generated LaTeX slides will be written",
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
        "--latent",
        default=None,
        help="Optional latent-space embeddings (TSV) to compare against the original matrix",
    )
    parser.add_argument(
        "--embedding",
        action="append",
        metavar="NAME=PATH",
        help="Latent embedding specification (repeatable; NAME must be unique)",
    )
    parser.add_argument(
        "--ordinations-output",
        default="benchmark_ordinations.pdf",
        help="Path where the PCA/t-SNE grid will be written when embeddings are provided",
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
    parser.set_defaults(matrix_log1p=True)
    parser.add_argument(
        "--frame-title",
        default=None,
        help="Optional title for the figure slide; defaults to the deck title",
    )
    parser.add_argument(
        "--table-title",
        default=None,
        help="Optional title for the summary table slide",
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
    renamed_results = _rename_results(results, renames)
    embeddings = _parse_embedding_specs(args.embedding, args.latent)
    if embeddings:
        _log(f"Loaded {len(embeddings)} embedding specification(s).")
    ordinations = _prepare_ordinations(args.matrix, args.matrix_log1p, embeddings, results)
    if ordinations:
        _log(f"Prepared ordinations for {len(ordinations)} dataset(s).")
    latent_dims = _collect_latent_dims(renamed_results)

    baseline = renames.get(args.baseline, args.baseline)
    if baseline and baseline not in renamed_results:
        raise SystemExit(
            f"Baseline '{baseline}' is not present in the loaded results after applying renames."
        )

    all_metric_keys = None
    for result in renamed_results.values():
        keys = set(result.mean_metrics.keys())
        all_metric_keys = keys if all_metric_keys is None else all_metric_keys & keys
    if not all_metric_keys:
        raise SystemExit("No common metrics found across the provided results.")

    requested_metrics: List[str] = []
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

    metric_tables: Dict[str, list[tuple[str, float, float]]] = {}
    for metric in requested_metrics:
        metric_tables[metric] = _collect_metric_rows(renamed_results, metric)

    _log(f"Rendering figures for metrics: {', '.join(requested_metrics)}.")
    figsize = _parse_figsize(args.figsize)
    figure_base = Path(args.figure_output)
    slides_path = Path(args.slides_output)
    slides_path.parent.mkdir(parents=True, exist_ok=True)

    figure_paths: Dict[str, Path] = {}
    for metric in requested_metrics:
        if len(requested_metrics) == 1:
            figure_path = figure_base
        else:
            if figure_base.suffix:
                figure_path = figure_base.with_name(f"{figure_base.stem}_{metric}{figure_base.suffix}")
            else:
                figure_path = figure_base / f"{metric}.pdf"
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        _log(f"Generating '{metric}' figure -> {figure_path}.")
        figure, _axes = plot_benchmark_figure(
            renamed_results,
            metric=metric,
            title=args.frame_title or args.title,
            baseline=baseline,
            figsize=figsize,
            output=str(figure_path),
        )

        figure_paths[metric] = figure_path

    try:  # Ensure resources are released when running headless
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:  # pragma: no cover - matplotlib already validated upstream
        pass

    figure_rel_paths: Dict[str, str] = {
        metric: os.path.relpath(path, start=slides_path.parent or Path("."))
        for metric, path in figure_paths.items()
    }
    ordination_rel: str | None = None
    if ordinations:
        ordination_path = Path(args.ordinations_output)
        ordination_path.parent.mkdir(parents=True, exist_ok=True)
        ord_title = args.frame_title or args.title or "PCA/t-SNE ordinations"
        ordination_order = _order_methods_by_metric(renamed_results, requested_metrics[0])
        _log(f"Generating ordination figure -> {ordination_path}.")
        ord_figure, _ord_axes = plot_ordination_grid(
            ordinations,
            title=ord_title,
            output=str(ordination_path),
            order=ordination_order,
            latent_dims=latent_dims,
        )
        try:
            import matplotlib.pyplot as plt

            plt.close(ord_figure)
        except ImportError:  # pragma: no cover
            pass
        ordination_rel = os.path.relpath(
            ordination_path, start=slides_path.parent or Path(".")
        )
    deck_title = _escape_latex(args.title)
    subtitle = _escape_latex(args.subtitle) if args.subtitle else None
    author = _escape_latex(args.author) if args.author else None
    date_text = args.date if args.date else ""
    frame_title = args.frame_title or args.title

    lines = [
        r"\documentclass{beamer}",
        rf"\usetheme{{{_escape_latex(args.theme)}}}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        "",
        rf"\title{{{deck_title}}}",
    ]
    if subtitle:
        lines.append(rf"\subtitle{{{subtitle}}}")
    if author:
        lines.append(rf"\author{{{author}}}")
    if date_text:
        lines.append(rf"\date{{{date_text}}}")
    else:
        lines.append(r"\date{}")

    lines.extend(
        [
            "",
            r"\begin{document}",
            r"\frame{\titlepage}",
        ]
    )

    multiple_metrics = len(requested_metrics) > 1
    for metric in requested_metrics:
        figure_rel = figure_rel_paths[metric]
        metric_title = _escape_latex(
            f"{frame_title}" if not multiple_metrics else f"{frame_title} — {metric}"
        )
        lines.extend(
            [
                "",
                rf"\begin{{frame}}{{{metric_title}}}",
                r"  \centering",
                rf"  \href{{run:{_escape_path(figure_rel)}}}{{\includegraphics[width=\textwidth]{{{_escape_path(figure_rel)}}}}}",
                r"  \vspace{1em}",
                rf"  \small Download: \href{{run:{_escape_path(figure_rel)}}}{{{_escape_latex(Path(figure_rel).name)}}}",
                r"\end{frame}",
            ]
        )

        table_title = args.table_title or f"{metric} summary"
        table_caption = _escape_latex(
            table_title if not multiple_metrics else f"{table_title} — {metric}"
        )
        lines.extend(
            [
                "",
                rf"\begin{{frame}}{{{table_caption}}}",
                r"  \centering",
                r"  \begin{tabular}{lrr}",
                r"    \toprule",
                r"    Method & Mean & Std \\",
                r"    \midrule",
            ]
        )

        for name, mean_value, std_value in metric_tables[metric]:
            escaped_name = _escape_latex(name)
            if baseline and name == baseline:
                escaped_name = rf"\textbf{{{escaped_name}}}"
            lines.append(
                rf"    {escaped_name} & {mean_value:.4g} & {std_value:.4g} \\",
            )

        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{frame}",
            ]
        )

    if ordination_rel:
        ord_title = _escape_latex(
            f"{frame_title} — Ordinations" if frame_title else "PCA/t-SNE ordinations"
        )
        lines.extend(
            [
                "",
                rf"\begin{{frame}}{{{ord_title}}}",
                r"  \centering",
                rf"  \href{{run:{_escape_path(ordination_rel)}}}{{\includegraphics[width=0.95\textwidth]{{{_escape_path(ordination_rel)}}}}}",
                r"  \vspace{1em}",
                rf"  \small Download: \href{{run:{_escape_path(ordination_rel)}}}{{{_escape_latex(Path(ordination_rel).name)}}}",
                r"\end{frame}",
            ]
        )

    lines.extend(["", r"\end{document}", ""])

    slides_path.write_text("\n".join(lines), encoding="utf-8")
    _log(f"Writing slides to {slides_path}.")

    for metric, path in figure_paths.items():
        print(f"Saved {metric} figure to {path.resolve()}")
    if ordination_rel:
        print(f"Saved ordination figure to {Path(args.ordinations_output).resolve()}")
    print(f"Saved slides to {slides_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
