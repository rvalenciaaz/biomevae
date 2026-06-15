"""CLI entry point for generating violin/box plots of reconstruction error distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from biomevae.reconstruction import CrossValResult

from ._recon_cli import dict_to_result, load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_figsize(value: str) -> tuple[float, float]:
    try:
        width_str, height_str = value.lower().split("x", 1)
        return float(width_str), float(height_str)
    except (ValueError, TypeError) as exc:
        raise SystemExit(
            "--figsize must be provided as WIDTHxHEIGHT (e.g. 10x5)."
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COMMON_METRICS = ("rmse", "mae", "r2", "cosine_similarity", "bray_curtis")


def _collect_fold_values(
    results: Mapping[str, CrossValResult],
    metric: str,
) -> Dict[str, list[float]]:
    """Extract per-fold values for *metric* from every method."""
    values: Dict[str, list[float]] = {}
    for name, result in results.items():
        fold_vals: list[float] = []
        for fold in result.fold_metrics:
            if metric not in fold:
                raise SystemExit(
                    f"Metric '{metric}' is missing from at least one fold of method '{name}'."
                )
            fold_vals.append(float(fold[metric]))
        values[name] = fold_vals
    return values


def _sort_methods_by_median(
    fold_values: Mapping[str, list[float]],
) -> list[str]:
    """Return method names sorted by median fold value (ascending)."""
    return sorted(
        fold_values.keys(),
        key=lambda name: float(np.median(fold_values[name])),
    )


def _plot_violin(
    fold_values: Mapping[str, list[float]],
    order: list[str],
    metric: str,
    baseline: str | None,
    figsize: tuple[float, float],
    output: str | None,
) -> object:
    """Create a violin + strip plot for a single metric and optionally save it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Prepare data in the requested order.
    data = [fold_values[name] for name in order]
    positions = list(range(len(order)))

    # Determine colours: baseline gets gray, others follow the colour cycle.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors: list[str] = []
    cycle_idx = 0
    for name in order:
        if baseline and name == baseline:
            colors.append("#999999")
        else:
            if prop_cycle:
                colors.append(prop_cycle[cycle_idx % len(prop_cycle)])
                cycle_idx += 1
            else:
                colors.append("#1f77b4")

    # Violin plot.
    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[idx])
        body.set_edgecolor("black")
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    # Strip plot overlay (individual fold points).
    rng = np.random.default_rng(42)
    for idx, name in enumerate(order):
        vals = np.asarray(fold_values[name])
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            idx + jitter,
            vals,
            color=colors[idx],
            edgecolors="black",
            linewidths=0.5,
            s=28,
            zorder=3,
            alpha=0.85,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Per-fold {metric} distribution")
    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "biomevae-recon-violin",
        description="Violin/box plots of per-fold reconstruction error distributions.",
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
            "Metric key to visualise (repeatable). When omitted, all common "
            "metrics found in every method are rendered."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path where the figure should be written",
    )
    parser.add_argument(
        "--figsize",
        default="10x5",
        help="Figure size specified as WIDTHxHEIGHT in inches (default: 10x5)",
    )
    parser.add_argument(
        "--baseline",
        default="nmf",
        help="Method name that should be highlighted in gray (default: nmf)",
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

    # Apply renames.
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
        # Baseline is optional; silently ignore if not present.
        baseline = None

    # Determine which metrics to plot.
    all_metric_keys: set[str] | None = None
    for result in renamed_results.values():
        keys = set(result.mean_metrics.keys())
        all_metric_keys = keys if all_metric_keys is None else all_metric_keys & keys
    if not all_metric_keys:
        raise SystemExit("No common metrics found across the provided results.")

    if args.metrics:
        requested_metrics: list[str] = []
        for metric in args.metrics:
            if metric not in all_metric_keys:
                raise SystemExit(
                    f"Metric '{metric}' is not present in all results; "
                    f"available: {sorted(all_metric_keys)}"
                )
            if metric not in requested_metrics:
                requested_metrics.append(metric)
    else:
        # Default: common metrics that are actually present, preserving a stable order.
        requested_metrics = [m for m in _COMMON_METRICS if m in all_metric_keys]
        if not requested_metrics:
            requested_metrics = sorted(all_metric_keys)

    if not requested_metrics:
        raise SystemExit("No metrics selected for plotting.")

    figsize = _parse_figsize(args.figsize)
    output_spec = args.output

    # Sort methods by median of the *first* requested metric.
    first_fold_values = _collect_fold_values(renamed_results, requested_metrics[0])
    order = _sort_methods_by_median(first_fold_values)

    for metric in requested_metrics:
        fold_values = _collect_fold_values(renamed_results, metric)

        # Determine per-metric output path.
        figure_output: str | None = None
        if output_spec:
            base = Path(output_spec)
            if len(requested_metrics) == 1:
                figure_output = str(base)
            else:
                if base.suffix:
                    figure_output = str(
                        base.with_name(f"{base.stem}_{metric}{base.suffix}")
                    )
                else:
                    figure_output = str(base / f"{metric}.png")

        fig = _plot_violin(
            fold_values,
            order=order,
            metric=metric,
            baseline=baseline,
            figsize=figsize,
            output=figure_output,
        )

        # Release figure resources.
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:  # pragma: no cover
            pass

        if figure_output:
            print(f"Saved {metric} violin plot to {Path(figure_output).resolve()}")
        else:
            print(f"Generated {metric} violin plot; use --output to save it.")


if __name__ == "__main__":  # pragma: no cover
    main()
