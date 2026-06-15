"""Cross-model SHAP comparison CLI.

Loads per-model ``otu_latent_summary.tsv`` artifacts produced by
``biomevae-interpret`` and generates consensus heatmaps, rank-agreement
matrices, and a consensus feature table.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_interpret_dir(spec: str) -> Tuple[str, str]:
    """Parse a ``NAME=PATH`` spec into *(name, path)*.

    Raises ``SystemExit`` on malformed input.
    """
    if "=" not in spec:
        raise SystemExit(
            f"--interpret-dir expects NAME=PATH, got: {spec!r}"
        )
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise SystemExit(f"Empty model name in --interpret-dir spec: {spec!r}")
    if not os.path.isdir(path):
        raise SystemExit(f"Interpret directory does not exist: {path}")
    return name, path


def _load_feature_importance(interpret_dir: str) -> pd.DataFrame:
    """Load ``otu_latent_summary.tsv`` and compute per-feature importance.

    Returns a DataFrame with columns ``feature`` and ``shap_mean_abs``
    (averaged across latent dimensions).
    """
    summary_path = os.path.join(interpret_dir, "otu_latent_summary.tsv")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing otu_latent_summary.tsv in {interpret_dir}")

    df = pd.read_csv(summary_path, sep="\t")

    # Accept both legacy column names (feature, shap_mean_abs) and current
    # names produced by biomevae-interpret (otu, mean_abs_shap).
    _COL_ALIASES = {"otu": "feature", "mean_abs_shap": "shap_mean_abs"}
    df.rename(columns={k: v for k, v in _COL_ALIASES.items() if k in df.columns}, inplace=True)

    required = {"feature", "shap_mean_abs"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"otu_latent_summary.tsv in {interpret_dir} is missing columns: "
            + ", ".join(sorted(missing))
        )

    agg = (
        df.groupby("feature", sort=False)["shap_mean_abs"]
        .mean()
        .reset_index()
    )
    return agg


def _parse_figsize(raw: str) -> Tuple[float, float]:
    """Parse a ``WIDTHxHEIGHT`` string into a *(w, h)* tuple."""
    parts = raw.lower().split("x")
    if len(parts) != 2:
        raise SystemExit(
            f"--figsize must be WIDTHxHEIGHT (e.g. 14x8), got: {raw!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise SystemExit(
            f"--figsize values must be numeric, got: {raw!r}"
        )


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _consensus_heatmap(
    importance: Dict[str, pd.DataFrame],
    top_k: int,
    output_prefix: str,
    figsize: Tuple[float, float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Determine the union of top-k features across all models.
    top_features: List[str] = []
    seen: set = set()
    for name, df in importance.items():
        ranked = df.sort_values("shap_mean_abs", ascending=False)
        for feat in ranked["feature"].iloc[:top_k]:
            if feat not in seen:
                top_features.append(feat)
                seen.add(feat)

    model_names = list(importance.keys())

    # Build the matrix (features × models), normalised per model.
    matrix = np.zeros((len(top_features), len(model_names)), dtype=np.float64)
    for j, name in enumerate(model_names):
        df = importance[name].set_index("feature")["shap_mean_abs"]
        col = np.array([df.get(f, 0.0) for f in top_features], dtype=np.float64)
        col_max = col.max()
        if col_max > 0:
            col = col / col_max
        matrix[:, j] = col

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=max(6, 10 - len(top_features) // 20))
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")
    ax.set_title("Consensus Feature Importance (normalised per model)")
    fig.colorbar(im, ax=ax, label="Normalised mean |SHAP|")
    fig.tight_layout()
    path = f"{output_prefix}_consensus_heatmap.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[interpret-compare] saved {path}")


def _rank_agreement(
    importance: Dict[str, pd.DataFrame],
    output_prefix: str,
    figsize: Tuple[float, float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr

    model_names = list(importance.keys())

    # Build a shared feature set (union of all features across models).
    all_features: set = set()
    for df in importance.values():
        all_features.update(df["feature"].tolist())
    all_features_sorted = sorted(all_features)

    # Rank vectors per model (missing features get worst rank).
    n_features = len(all_features_sorted)
    rank_vectors: Dict[str, np.ndarray] = {}
    for name, df in importance.items():
        ranked = df.sort_values("shap_mean_abs", ascending=False)
        rank_map = {f: r + 1 for r, f in enumerate(ranked["feature"])}
        worst = len(rank_map) + 1
        vec = np.array(
            [rank_map.get(f, worst) for f in all_features_sorted],
            dtype=np.float64,
        )
        rank_vectors[name] = vec

    n = len(model_names)
    corr_matrix = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(rank_vectors[model_names[i]], rank_vectors[model_names[j]])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_names)
    # Annotate cells with correlation values.
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{corr_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if abs(corr_matrix[i, j]) > 0.6 else "black",
            )
    ax.set_title("Feature-Rank Agreement (Spearman)")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    path = f"{output_prefix}_rank_agreement.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[interpret-compare] saved {path}")


def _consensus_table(
    importance: Dict[str, pd.DataFrame],
    top_k: int,
    output_prefix: str,
) -> None:
    """Write a TSV of features appearing in the top-k of multiple models."""
    model_names = list(importance.keys())
    n_models = len(model_names)

    # Collect top-k feature sets per model and per-model importance.
    topk_sets: Dict[str, set] = {}
    importance_maps: Dict[str, Dict[str, float]] = {}
    for name, df in importance.items():
        ranked = df.sort_values("shap_mean_abs", ascending=False)
        topk_sets[name] = set(ranked["feature"].iloc[:top_k])
        importance_maps[name] = dict(
            zip(ranked["feature"], ranked["shap_mean_abs"])
        )

    # Union of all top-k features.
    all_topk = set()
    for s in topk_sets.values():
        all_topk.update(s)

    rows = []
    for feat in sorted(all_topk):
        n_in = sum(1 for s in topk_sets.values() if feat in s)
        avg_shap = sum(
            importance_maps[m].get(feat, 0.0) for m in model_names
        ) / n_models
        rows.append(
            {"feature": feat, "n_models_topk": n_in, "avg_shap_mean_abs": avg_shap}
        )

    result = pd.DataFrame(rows).sort_values(
        ["n_models_topk", "avg_shap_mean_abs"], ascending=[False, False]
    )
    path = f"{output_prefix}_consensus_features.tsv"
    result.to_csv(path, sep="\t", index=False)
    print(f"[interpret-compare] saved {path}")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "biomevae-interpret-compare",
        description=(
            "Compare feature-importance rankings across multiple "
            "biomevae-interpret runs."
        ),
    )
    parser.add_argument(
        "--interpret-dir",
        action="append",
        required=True,
        metavar="NAME=PATH",
        dest="interpret_dirs",
        help=(
            "Model interpret output directory in NAME=PATH format. "
            "May be repeated for each model to compare."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features per model to include (default: 20)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path prefix for output figures and tables",
    )
    parser.add_argument(
        "--figsize",
        default="14x8",
        help="Figure size as WIDTHxHEIGHT (default: 14x8)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    figsize = _parse_figsize(args.figsize)

    # Parse and load all model importance data.
    importance: Dict[str, pd.DataFrame] = {}
    for spec in args.interpret_dirs:
        name, path = _parse_interpret_dir(spec)
        if name in importance:
            raise SystemExit(f"Duplicate model name: {name!r}")
        importance[name] = _load_feature_importance(path)

    if len(importance) < 2:
        raise SystemExit(
            "At least two --interpret-dir entries are required for comparison."
        )

    # Ensure the output directory exists.
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate artifacts.
    _consensus_heatmap(importance, args.top_k, args.output, figsize)
    _rank_agreement(importance, args.output, figsize)
    _consensus_table(importance, args.top_k, args.output)

    print("[interpret-compare] done.")
