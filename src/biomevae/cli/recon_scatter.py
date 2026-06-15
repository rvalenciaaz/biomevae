"""CLI entry point for observed vs predicted scatter plots from biomevae model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from biomevae.data import load_matrix


def _parse_figsize(value: str) -> Tuple[float, float]:
    try:
        width_str, height_str = value.lower().split("x", 1)
        return float(width_str), float(height_str)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "--figsize must be WIDTHxHEIGHT (e.g. 12x5)."
        ) from exc


def _parse_recon_specs(specs: List[str]) -> Dict[str, Path]:
    """Parse NAME=PATH reconstruction specs into an ordered dict."""
    recons: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"Invalid --recon spec '{spec}'. Use NAME=PATH format "
                "(e.g. base=output/base/recon.tsv)."
            )
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit("Reconstruction entry name cannot be empty.")
        if name in recons:
            raise SystemExit(f"Duplicate reconstruction name: {name}")
        recons[name] = Path(raw_path).expanduser()
    if not recons:
        raise SystemExit("No reconstructions provided. Use --recon NAME=PATH at least once.")
    return recons


def _load_recon(path: Path) -> np.ndarray:
    """Load a recon.tsv (samples x features, sample index in first column)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.to_numpy(dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="biomevae-recon-scatter",
        description="Generate observed vs predicted scatter plots from biomevae reconstructions.",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Original counts TSV (rows=taxa, cols=[clade_name, NCBI_tax_id, sample1, ...]).",
    )
    ap.add_argument(
        "--recon",
        action="append",
        default=[],
        help=(
            "Repeatable NAME=PATH spec for each model's recon.tsv. "
            "Example: --recon base=runs/base/recon.tsv --recon graph=runs/graph/recon.tsv"
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the figure (e.g. scatter.png).",
    )
    ap.add_argument(
        "--figsize",
        type=_parse_figsize,
        default=(12, 5),
        help="Figure size as WIDTHxHEIGHT (default: 12x5).",
    )
    ap.add_argument(
        "--log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply log1p transform to counts (default: True). Use --no-log1p to disable.",
    )
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=0.05,
        help="Fraction of flattened points to plot (default: 0.05).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for point subsampling (default: 42).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = build_parser().parse_args(argv)

    recons = _parse_recon_specs(args.recon)
    n_models = len(recons)

    # Load the original counts matrix: [samples, features]
    X_obs, sample_names = load_matrix(args.input, log1p=args.log1p)

    fig_w, fig_h = args.figsize
    fig, axes = plt.subplots(1, n_models, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.ravel()

    rng = np.random.RandomState(args.seed)

    for idx, (name, recon_path) in enumerate(recons.items()):
        ax = axes[idx]

        X_recon = _load_recon(recon_path)

        if args.log1p:
            X_recon = np.log1p(X_recon).astype(np.float32)

        # Ensure shapes match
        if X_obs.shape != X_recon.shape:
            raise SystemExit(
                f"Shape mismatch for '{name}': observed {X_obs.shape} vs "
                f"reconstructed {X_recon.shape}."
            )

        obs_flat = X_obs.ravel()
        pred_flat = X_recon.ravel()

        # Subsample for plotting
        n_total = len(obs_flat)
        n_plot = max(1, int(n_total * args.sample_frac))
        if n_plot < n_total:
            indices = rng.choice(n_total, size=n_plot, replace=False)
            obs_plot = obs_flat[indices]
            pred_plot = pred_flat[indices]
        else:
            obs_plot = obs_flat
            pred_plot = pred_flat

        # Compute R^2 and RMSE on the full data
        ss_res = np.sum((obs_flat - pred_flat) ** 2)
        ss_tot = np.sum((obs_flat - np.mean(obs_flat)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = np.sqrt(np.mean((obs_flat - pred_flat) ** 2))

        # Scatter
        ax.scatter(obs_plot, pred_plot, alpha=0.15, s=4, edgecolors="none")

        # Identity line
        lo = min(obs_plot.min(), pred_plot.min())
        hi = max(obs_plot.max(), pred_plot.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y = x")

        # Annotation
        ax.text(
            0.05,
            0.92,
            f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(name)

    fig.tight_layout()

    # Ensure output parent directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved reconstruction scatter plot to {args.output}")


if __name__ == "__main__":
    main()
