"""CLI entry point for exporting pairwise statistical significance tables and heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

from biomevae.reconstruction import (
    CrossValResult,
    compute_pairwise_metric_stats,
    compute_pairwise_seed_stats,
    adjust_pvalues_bh,
    adjust_pvalues_bonferroni,
)

from ._recon_cli import dict_to_result, load_json


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


def _parse_figsize(value: str) -> tuple[float, float]:
    try:
        width_str, height_str = value.lower().split("x", 1)
        return float(width_str), float(height_str)
    except (ValueError, TypeError) as exc:
        raise SystemExit(
            "--figsize must be provided as WIDTHxHEIGHT (e.g. 8x7)."
        ) from exc


def _significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def _build_pvalue_matrix(
    comparisons: list[dict],
    methods: list[str],
) -> dict[tuple[str, str], float]:
    """Build a lookup from (model_a, model_b) -> p_value for the matrix."""
    lookup: dict[tuple[str, str], float] = {}
    for row in comparisons:
        a, b = row["model_a"], row["model_b"]
        lookup[(a, b)] = float(row["p_value"])
        lookup[(b, a)] = float(row["p_value"])
    return lookup


def _write_tsv(
    comparisons: list[dict],
    output_path: Path,
    *,
    test_name: str,
) -> None:
    """Write the full pairwise comparison table as a TSV file."""
    base_header = [
        "model_a",
        "model_b",
        "mean_diff",
        "median_diff",
        "n",
        "n_positive",
        "n_negative",
        "p_value",
        "p_value_bh",
        "p_value_bonferroni",
    ]
    extra_keys = ["p_value_sign", "p_value_wilcoxon", "p_value_tcorrected"]
    header = list(base_header)
    if test_name == "seed":
        header += [key for key in extra_keys if key not in header]
    lines = ["\t".join(header)]
    for row in comparisons:
        values = [
            str(row["model_a"]),
            str(row["model_b"]),
            f"{float(row['mean_diff']):.6g}",
            f"{float(row['median_diff']):.6g}",
            str(row["n"]),
            str(row["n_positive"]),
            str(row["n_negative"]),
            f"{float(row['p_value']):.6g}",
            f"{float(row['p_value_bh']):.6g}",
            f"{float(row['p_value_bonferroni']):.6g}",
        ]
        if test_name == "seed":
            values.extend(f"{float(row[key]):.6g}" for key in extra_keys)
        lines.append("\t".join(values))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(
    comparisons: list[dict],
    methods: list[str],
    output_path: Path,
) -> None:
    """Write a methods x methods p-value matrix as a LaTeX table."""
    lookup = _build_pvalue_matrix(comparisons, methods)
    n = len(methods)
    col_spec = "l" + "c" * n

    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(methods) + r" \\")
    lines.append(r"\midrule")
    for a in methods:
        cells = []
        for b in methods:
            if a == b:
                cells.append("---")
            elif (a, b) in lookup:
                p = lookup[(a, b)]
                stars = _significance_stars(p)
                cells.append(f"{p:.3g}{stars}")
            else:
                cells.append("")
        lines.append(a + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_heatmap(
    comparisons: list[dict],
    methods: list[str],
    output_path: Path,
    figsize: tuple[float, float],
    *,
    test_label: str,
) -> None:
    """Generate a heatmap figure of the p-value matrix (log scale)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    lookup = _build_pvalue_matrix(comparisons, methods)
    n = len(methods)

    matrix = np.full((n, n), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if a != b and (a, b) in lookup:
                matrix[i, j] = lookup[(a, b)]

    # Replace exact zeros to avoid log(0)
    min_nonzero = np.nanmin(matrix[matrix > 0]) if np.any(matrix > 0) else 1e-300
    matrix = np.where(
        (matrix == 0) & ~np.isnan(matrix),
        min_nonzero * 0.1,
        matrix,
    )

    fig, ax = plt.subplots(figsize=figsize)

    vmin = np.nanmin(matrix[~np.isnan(matrix)]) if np.any(~np.isnan(matrix)) else 1e-10
    vmax = 1.0
    if vmin >= vmax:
        vmin = vmax * 1e-3

    im = ax.imshow(
        matrix,
        cmap="RdYlGn",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, label="p-value")  # noqa: F841

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)

    # Add significance stars as text annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                stars = _significance_stars(matrix[i, j])
                if stars:
                    ax.text(
                        j,
                        i,
                        stars,
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                        fontweight="bold",
                    )

    ax.set_title(f"Pairwise p-values ({test_label})")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "biomevae-pairwise-table",
        description="Export pairwise statistical significance tables and heatmaps.",
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more JSON files produced by biomevae-allcomp",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        metavar="KEY",
        help=(
            "Metric key to include (repeatable). Defaults to rmse and mae when omitted."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path prefix (files will be named {output}_{metric}_pairwise.{ext})",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["tsv", "latex", "both"],
        default="both",
        help="Output format: tsv, latex, or both (default: both)",
    )
    parser.add_argument(
        "--figsize",
        default="8x7",
        help="Figure size specified as WIDTHxHEIGHT in inches (default: 8x7)",
    )
    parser.add_argument(
        "--test",
        choices=["seed", "fold"],
        default="seed",
        help=(
            "Statistical test used for pairwise comparisons.  ``seed`` "
            "(the default) runs a paired Wilcoxon signed-rank test plus "
            "a Nadeau--Bengio corrected paired t-test on the per-seed "
            "mean metrics stored in ``metadata['per_seed_mean_metrics']``.  "
            "``fold`` falls back to the legacy sign test on fold-level "
            "metrics and is only meaningful when every method was "
            "evaluated on identical fold partitions."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help=(
            "Train-set fraction used by the Nadeau--Bengio correction "
            "(seed test only).  Should match the ``train_fraction`` "
            "passed to the cross-validation helpers.  Default: 0.9."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    results = _load_results(args.input)
    print(f"Loaded {len(results)} method(s) from {len(args.input)} input file(s).")

    metrics = args.metrics if args.metrics else ["rmse", "mae"]
    figsize = _parse_figsize(args.figsize)
    output_prefix = args.output
    fmt = args.fmt
    test_name = args.test
    test_labels = {
        "seed": "Nadeau-Bengio corrected paired t",
        "fold": "sign test",
    }
    test_label = test_labels[test_name]

    for metric in metrics:
        # Verify metric is present in all results
        for name, result in results.items():
            if metric not in result.mean_metrics:
                raise SystemExit(
                    f"Metric '{metric}' is not present in results for method '{name}'. "
                    f"Available: {sorted(result.mean_metrics.keys())}"
                )

        if test_name == "seed":
            comparisons = compute_pairwise_seed_stats(
                results, metric, train_fraction=float(args.train_fraction),
            )
        else:
            comparisons = compute_pairwise_metric_stats(results, metric)
        if not comparisons:
            print(f"No pairwise comparisons generated for '{metric}'; skipping.")
            continue

        # Apply multiple-testing corrections to the canonical p-value.
        raw_pvalues = [float(row["p_value"]) for row in comparisons]
        pvals_bh = adjust_pvalues_bh(raw_pvalues)
        pvals_bonferroni = adjust_pvalues_bonferroni(raw_pvalues)
        for index, row in enumerate(comparisons):
            row["p_value_bh"] = pvals_bh[index]
            row["p_value_bonferroni"] = pvals_bonferroni[index]

        methods = sorted(results.keys())

        # TSV
        if fmt in ("tsv", "both"):
            tsv_path = Path(f"{output_prefix}_{metric}_pairwise.tsv")
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            _write_tsv(comparisons, tsv_path, test_name=test_name)
            print(f"Wrote TSV: {tsv_path.resolve()}")

        # LaTeX
        if fmt in ("latex", "both"):
            tex_path = Path(f"{output_prefix}_{metric}_pairwise.tex")
            tex_path.parent.mkdir(parents=True, exist_ok=True)
            _write_latex(comparisons, methods, tex_path)
            print(f"Wrote LaTeX: {tex_path.resolve()}")

        # Heatmap (always generated)
        pdf_path = Path(f"{output_prefix}_{metric}_pairwise.pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        _write_heatmap(
            comparisons, methods, pdf_path, figsize, test_label=test_label,
        )
        print(f"Wrote heatmap: {pdf_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
