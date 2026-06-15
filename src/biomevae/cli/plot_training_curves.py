#!/usr/bin/env python3
"""Plot training and validation curves from training_log.tsv files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training curves from one or more biomevae training_log.tsv files."
        )
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help=(
            "Repeatable NAME=PATH mapping for each model's training_log.tsv. "
            "Example: --log base=runs/base/training_log.tsv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory for the plots. Defaults to the current directory."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--metric",
        default="loss",
        help="Metric suffix to plot (default: loss -> train_loss/val_loss columns).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of epochs to plot (default: 50).",
    )
    parser.add_argument(
        "--yscale",
        choices=("log", "linear"),
        default="log",
        help=(
            "Y-axis scale (default: log). Log scale keeps models with vastly "
            "different magnitudes (e.g. NB-NLL based PhILR-VAE / TreeNB-VAE "
            "alongside MSE-based VAEs) visible in the same panel."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args(argv)


def _style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)


def _read_log(label: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Training log not found for {label}: {path}")
    return pd.read_csv(path, sep="\t")


def _parse_logs(log_args: list[str]) -> dict[str, Path]:
    logs: dict[str, Path] = {}
    for item in log_args:
        if "=" not in item:
            raise SystemExit(
                "Each --log entry must be in NAME=PATH form (e.g., base=path/to/training_log.tsv)."
            )
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit("Log entry name cannot be empty.")
        if name in logs:
            raise SystemExit(f"Duplicate log name detected: {name}")
        logs[name] = Path(raw_path).expanduser()
    if not logs:
        raise SystemExit(
            "No logs provided. Use --log NAME=PATH at least once."
        )
    return logs


def _configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
        }
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logs = _parse_logs(args.log)
    metric = args.metric
    train_col = f"train_{metric}"
    val_col = f"val_{metric}"
    output_dir = args.output if args.output is not None else Path(".")
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_matplotlib()

    colors = plt.cm.tab10.colors

    def plot_panel(kind: str, ylabel: str, filename: str, use_train: bool, use_val: bool) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        max_epoch = 0
        plotted_any = False
        for idx, (label, path) in enumerate(logs.items()):
            df = _read_log(label, path)
            if args.max_epochs is not None:
                df = df[df["epoch"] <= args.max_epochs]
            if not df.empty:
                max_epoch = max(max_epoch, int(df["epoch"].max()))
            color = colors[idx % len(colors)]
            missing_cols = [
                col for col, needed in ((train_col, use_train), (val_col, use_val))
                if needed and col not in df.columns
            ]
            if missing_cols:
                print(
                    f"  WARNING: {label}: skipping — training_log.tsv is "
                    f"missing column(s) {missing_cols} required for "
                    f"metric '{metric}'."
                )
                continue
            if use_train:
                ax.plot(
                    df["epoch"],
                    df[train_col],
                    label=f"{label} (train)",
                    color=color,
                    linewidth=2.0,
                )
                plotted_any = True
            if use_val:
                ax.plot(
                    df["epoch"],
                    df[val_col],
                    label=f"{label} (val)",
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                )
                plotted_any = True
        _style_axis(ax, ylabel)
        if max_epoch > 0:
            target_ticks = 6
            nbins = min(max_epoch, target_ticks)
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=nbins, min_n_ticks=4)
            )
        if args.yscale == "log":
            # Log scale accommodates NB-NLL-based models (PhILR-VAE,
            # TreeNB-VAE) whose recon loss lives in the hundreds/thousands
            # range alongside MSE-based VAEs whose loss lives near zero.
            # Without it the small-scale curves collapse to the x-axis and
            # the NB models dominate the plot.
            ax.set_yscale("log")
        ax.set_title(kind)
        if plotted_any:
            ax.legend(ncol=2)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=150)
        print(f"Saved {kind.lower()} plot to {output_path}")
        if args.show:
            plt.show()

    title_suffix = f" ({args.title})" if args.title else ""
    label = f"{metric.upper()} curves{title_suffix}"
    plot_panel(
        kind=f"Training {label}",
        ylabel=f"Train {metric}",
        filename=f"training_{metric}_curves.png",
        use_train=True,
        use_val=False,
    )
    plot_panel(
        kind=f"Validation {label}",
        ylabel=f"Validation {metric}",
        filename=f"validation_{metric}_curves.png",
        use_train=False,
        use_val=True,
    )
    plot_panel(
        kind=f"Training + validation {label}",
        ylabel=metric.capitalize(),
        filename=f"train_val_{metric}_curves.png",
        use_train=True,
        use_val=True,
    )


if __name__ == "__main__":
    main()
