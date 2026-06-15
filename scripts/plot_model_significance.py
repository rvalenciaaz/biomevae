#!/usr/bin/env python3
"""Statistical comparison of models across the single-study benchmark.

Implements the standard Demšar (2006) procedure for comparing multiple
classifiers over multiple datasets:

  1. Friedman omnibus test on per-study model ranks (are the models
     distinguishable at all?).
  2. If significant, a Nemenyi post-hoc critical-difference (CD) diagram:
     models connected by a bar are NOT significantly different.

The block design (study = block, model = treatment) controls for the large
between-study spread, which a naive per-metric ranking ignores.

Outputs (to results/figures/current_results/ by default):
    model_significance_cd.{pdf,png}
    model_rank_stats.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_current_results import model_color, model_label  # noqa: E402

# Nemenyi two-tailed q_0.05 critical values (studentized range / sqrt(2)),
# indexed by number of models k. Standard table (Demšar 2006).
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("results/figures/current_results/latest_single_meta_summary.tsv"),
    )
    p.add_argument("--metric", default="auroc", choices=["auroc", "balanced_accuracy", "f1_macro"])
    p.add_argument("--outdir", type=Path, default=Path("results/figures/current_results"))
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compute_ranks(meta: pd.DataFrame, metric: str):
    meta = meta.copy()
    meta[metric] = pd.to_numeric(meta[metric], errors="coerce")
    piv = meta.pivot_table(index="study", columns="model_key", values=metric, aggfunc="mean")
    piv = piv.dropna(axis=1)  # complete-case models only (Friedman needs full blocks)
    # Higher metric = better = lower (better) rank.
    ranks = piv.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean(axis=0).sort_values()
    stat, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    return piv, avg_rank, {"friedman_chi2": float(stat), "friedman_p": float(p), "k": piv.shape[1], "N": piv.shape[0]}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    q = NEMENYI_Q05[k]
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def cliques(avg_rank: pd.Series, cd: float):
    """Maximal groups of models whose mean ranks differ by < CD."""
    models = list(avg_rank.index)
    vals = avg_rank.to_numpy()
    bars = []
    for i in range(len(models)):
        j = i
        while j + 1 < len(models) and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i:
            bars.append((i, j))
    # drop bars fully contained in another
    pruned = [b for b in bars if not any(a[0] <= b[0] and b[1] <= a[1] and a != b for a in bars)]
    return [(vals[i], vals[j]) for i, j in pruned]


def plot_cd(avg_rank: pd.Series, cd: float, info: dict, metric: str, outdir: Path, formats,
            stem: str = "model_significance_cd", unit: str = "studies"):
    import matplotlib.pyplot as plt

    k = len(avg_rank)
    models = list(avg_rank.index)
    ranks = avg_rank.to_numpy()
    lo, hi = 1, k

    fig, ax = plt.subplots(figsize=(13, 0.40 * k + 1.4), constrained_layout=True)
    ax.set_xlim(lo - 0.5, hi + 0.5)  # rank 1 (best) on the left, k (worst) on the right
    # top axis with rank ticks
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_position(("axes", 1.0))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks([])
    ax.set_xticks(range(lo, hi + 1))
    ax.set_xlabel(f"Average rank ({metric.replace('_', ' ')}; 1 = best)")
    ax.tick_params(axis="x", length=4)

    top_y = 0.78
    # CD bar indicator
    ax.plot([lo, lo + cd], [0.93, 0.93], color="#111111", lw=2.4)
    ax.plot([lo, lo], [0.915, 0.945], color="#111111", lw=1.4)
    ax.plot([lo + cd, lo + cd], [0.915, 0.945], color="#111111", lw=1.4)
    ax.text(lo + cd / 2, 0.965, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    n_left = (k + 1) // 2
    label_xs_left = lo - 0.45
    label_xs_right = hi + 0.45
    for idx, (m, r) in enumerate(zip(models, ranks)):
        connector_y = top_y - 0.045 * (idx if idx < n_left else (k - 1 - idx))
        if idx < n_left:
            lx, ha = label_xs_left, "right"
        else:
            lx, ha = label_xs_right, "left"
        ax.plot([r, r], [top_y + 0.02, connector_y], color=model_color(m), lw=1.6)
        ax.plot([r, lx], [connector_y, connector_y], color=model_color(m), lw=1.6)
        ax.text(
            lx + (-0.06 if ha == "right" else 0.06),
            connector_y,
            f"{model_label(m)}  ({r:.1f})",
            ha=ha,
            va="center",
            fontsize=9.5,
            color="#1f2937",
        )

    # clique bars (non-significant groups)
    bar_y = top_y + 0.06
    for a, b in cliques(avg_rank, cd):
        ax.plot([a, b], [bar_y, bar_y], color="#6b7280", lw=4.5, solid_capstyle="round")
        bar_y += 0.035

    ptxt = "p < 0.001" if info["friedman_p"] < 1e-3 else f"p = {info['friedman_p']:.3f}"
    ax.set_title(
        f"Model comparison across {info['N']} {unit} — Friedman χ² = {info['friedman_chi2']:.1f}, "
        f"{ptxt}; Nemenyi CD diagram (connected = not sig. different)",
        fontsize=12,
        pad=26,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else 1200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    meta = pd.read_csv(args.meta, sep="\t")
    piv, avg_rank, info = compute_ranks(meta, args.metric)
    cd = nemenyi_cd(info["k"], info["N"])
    info["nemenyi_cd_0.05"] = cd

    out = avg_rank.rename("avg_rank").reset_index().rename(columns={"index": "model_key"})
    out["model_label"] = out["model_key"].map(model_label)
    out.to_csv(args.outdir / "model_rank_stats.tsv", sep="\t", index=False)

    paths = plot_cd(avg_rank, cd, info, args.metric, args.outdir, args.formats)
    print("Friedman / Nemenyi:", info)
    print(out.to_string(index=False))
    print("\nWrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
