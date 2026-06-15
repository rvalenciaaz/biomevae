"""CLI entry point for generating per-taxonomy-level metric breakdown bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

from biomevae.reconstruction import CrossValResult

from ._recon_cli import dict_to_result, load_json


_DEFAULT_LEVELS = ("phylum", "class", "order", "family", "genus", "species")


def _parse_figsize(value: str) -> tuple[float, float]:
    try:
        width_str, height_str = value.lower().split("x", 1)
        return float(width_str), float(height_str)
    except (ValueError, TypeError) as exc:
        raise SystemExit(
            "--figsize must be provided as WIDTHxHEIGHT (e.g. 12x6)."
        ) from exc


def _parse_renames(items: list[str] | None) -> Dict[str, str]:
    renames: Dict[str, str] = {}
    if not items:
        return renames
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid rename '{item}'. Use OLD=NEW format.")
        old, new = (part.strip() for part in item.split("=", 1))
        if not old or not new:
            raise SystemExit("Rename assignments must include both old and new labels.")
        renames[old] = new
    return renames


def _load_results(paths: list[str]) -> Dict[str, CrossValResult]:
    results: Dict[str, CrossValResult] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise SystemExit(
                f"Top-level structure in '{path}' must be a JSON object "
                "mapping method names to results."
            )
        for name, raw in payload.items():
            if not isinstance(raw, Mapping):
                raise SystemExit(
                    f"Entry '{name}' in '{path}' must be an object "
                    "describing the metrics."
                )
            if name in results:
                raise SystemExit(
                    f"Method '{name}' appears multiple times across inputs; "
                    "rename duplicates first."
                )
            results[name] = dict_to_result(raw)
    if not results:
        raise SystemExit("No methods found in the provided inputs.")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "biomevae-hierarchy-figure",
        description="Generate per-taxonomy-level metric breakdown bar charts.",
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more JSON files produced by biomevae-allcomp --taxonomy",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Base metric name to visualise (e.g. 'rmse', 'mae', 'r2')",
    )
    parser.add_argument(
        "--levels",
        default=None,
        nargs="+",
        help=(
            "Taxonomy levels to include on the x-axis "
            "(default: phylum class order family genus species)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path where the figure should be saved",
    )
    parser.add_argument(
        "--figsize",
        default="12x6",
        help="Figure size as WIDTHxHEIGHT in inches (default: 12x6)",
    )
    parser.add_argument(
        "--baseline",
        default="nmf",
        help="Method name to highlight as the baseline (default: nmf)",
    )
    parser.add_argument(
        "--rename",
        nargs="*",
        metavar="OLD=NEW",
        help="Rename method labels before plotting (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    results = _load_results(args.input)
    renames = _parse_renames(args.rename)

    # Apply renames --------------------------------------------------------
    renamed_results: Dict[str, CrossValResult] = {}
    for name, result in results.items():
        new_name = renames.get(name, name)
        if new_name in renamed_results:
            raise SystemExit(
                f"Renaming results would create duplicate label '{new_name}'."
            )
        renamed_results[new_name] = result

    baseline = renames.get(args.baseline, args.baseline)
    metric = args.metric
    levels = list(args.levels) if args.levels else list(_DEFAULT_LEVELS)
    figsize = _parse_figsize(args.figsize)

    # Validate that the requested metric keys exist ------------------------
    method_names = list(renamed_results.keys())
    for level in levels:
        key = f"{metric}_{level}"
        for name in method_names:
            if key not in renamed_results[name].mean_metrics:
                raise SystemExit(
                    f"Metric key '{key}' not found for method '{name}'. "
                    f"Available keys: {sorted(renamed_results[name].mean_metrics.keys())}"
                )

    # Sort methods: baseline first, then alphabetical ----------------------
    ordered_methods: list[str] = []
    if baseline and baseline in renamed_results:
        ordered_methods.append(baseline)
    for name in sorted(renamed_results.keys()):
        if name not in ordered_methods:
            ordered_methods.append(name)

    # Lazy import of matplotlib --------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    n_levels = len(levels)
    n_methods = len(ordered_methods)
    x = np.arange(n_levels)
    total_width = 0.8
    bar_width = total_width / n_methods

    fig, ax = plt.subplots(figsize=figsize)

    for idx, method in enumerate(ordered_methods):
        result = renamed_results[method]
        means = [result.mean_metrics[f"{metric}_{level}"] for level in levels]
        stds = [result.std_metrics.get(f"{metric}_{level}", 0.0) for level in levels]
        offset = (idx - (n_methods - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            label=method,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([level.capitalize() for level in levels])
    ax.set_xlabel("Taxonomy Level")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by Taxonomy Level")
    ax.legend()
    fig.tight_layout()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300)
        png_path = output_path.with_suffix(".png")
        if png_path != output_path:
            fig.savefig(str(png_path), dpi=300)
            print(f"Saved figure to {png_path.resolve()}")
        print(f"Saved figure to {output_path.resolve()}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
