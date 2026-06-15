#!/usr/bin/env python3
"""Render current biomevae result summaries as multi-panel figures.

The script consumes the repository-level result tables produced by the
single-study and strict LOSO workflows:

    results/figures/current_results/latest_single_meta_summary.tsv
    results/figures/current_results/latest_loso_strict_summary.tsv

It writes publication-style PDF and PNG figures plus compact aggregate
tables to results/figures/current_results/ by default.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Study (cohort) disease-type annotation. Shared with
# scripts/plot_disease_category_stats.py so the heatmap colour strip uses the
# same authoritative mapping (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from plot_disease_category_stats import (  # noqa: E402
        STUDY_DISEASE,
        CATEGORY_GROUP,
        CATEGORY_COLORS,
    )
except Exception:  # pragma: no cover - keep the figure script usable standalone
    STUDY_DISEASE, CATEGORY_GROUP, CATEGORY_COLORS = {}, {}, {}

# Colour used for cohorts with no disease-category assignment.
UNKNOWN_TYPE_COLOR = "#cccccc"


METRIC_LABELS = {
    "balanced_accuracy": "Balanced accuracy",
    "f1_macro": "Macro F1",
    "auroc": "AUROC",
    "rmse": "RMSE",
    "mae": "MAE",
}

MODEL_LABELS = {
    "beta-vae": "Beta-VAE",
    "diva-beta-vae": "DIVA Beta-VAE",
    "diva-hyp-philr-nb": "DIVA Hyp-PhILR-NB",
    "diva-tree-dtm-vae": "DIVA TreeDTM-VAE",
    "diva-treenbvae": "DIVA TreeNB-VAE",
    "tree-dtm-vae": "TreeDTM-VAE",
    "phylodiva-beta-vae": "PhyloDIVA Beta-VAE",
    "phylodiva-hyp-philr-nb": "PhyloDIVA Hyp-PhILR-NB",
    "phylodiva-tree-dtm-vae": "PhyloDIVA TreeDTM-VAE",
    "taxi-hyp-philrvae": "TAXI Hyp-PhILR-NB",
    "taxi-tree-dtm-vae": "TAXI TreeDTM-VAE",
    "dsvae-sup": "DSVAE supervised",
    "dsvae-unsup": "DSVAE unsupervised",
    "fuse-vae": "PhyloFusion VAE",
    "graph-vae": "Graph VAE",
    "hyp-philr-zinb": "Hyp-PhILR-ZINB",
    "hyp-philrvae": "Hyp-PhILR-NB",
    "hyp-tax-vae": "Hyp+Tax VAE",
    "hyp-vae": "Hyperbolic VAE",
    "philr-gauss-vae": "PhILR Gaussian VAE",
    "philrvae": "PhILR-NB VAE",
    "tax-vae": "Tax-aware VAE",
    "treedirichlet-vae": "TreeDirichlet VAE",
    "treenbvae": "TreeNB-VAE",
    "treeprior-vae": "TreePrior VAE",
    "vanilla-vae": "Vanilla VAE",
    "capda-vae": "CAPDA-VAE",
    "nmf": "NMF",
    "xgb-baseline": "XGBoost SGB",
    "xgboost-baseline": "XGBoost SGB",
    "xgb-coral": "XGBoost SGB + CORAL",
}

# Paul Tol's "muted" qualitative scheme (colour-blind-safe and print-friendly)
# plus a neutral grey for the baseline. Avoids the red/green family clash that
# Nature flags for accessibility.
FAMILY_COLORS = {
    "baseline": "#111111",  # near-black – non-VAE reference, max luminance separation
    "vanilla": "#332288",   # indigo
    "tree": "#117733",      # green
    "philr": "#88CCEE",     # cyan
    "hyperbolic": "#AA4499",  # purple
    "taxonomy": "#DDCC77",  # sand
    "graph": "#CC6677",     # rose
    "diva": "#882255",      # wine
    "dsvae": "#44AA99",     # teal
    "fusion": "#999933",    # olive
}

FAMILY_LABELS = {
    "baseline": "Baseline (XGBoost)",
    "vanilla": "Vanilla VAE",
    "tree": "Tree-structured",
    "philr": "PhILR",
    "hyperbolic": "Hyperbolic",
    "taxonomy": "Taxonomy-aware",
    "graph": "Graph",
    "diva": "DIVA",
    "dsvae": "DSVAE",
    "fusion": "PhyloFusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot current biomevae single-study and strict LOSO results."
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("results/figures/current_results/latest_single_meta_summary.tsv"),
        help="Single-study meta summary TSV.",
    )
    parser.add_argument(
        "--loso",
        type=Path,
        default=Path("results/figures/current_results/latest_loso_strict_summary.tsv"),
        help="Strict LOSO summary TSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/figures/current_results"),
        help="Directory for figures and aggregate tables.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats to write.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Poster-grade defaults: larger, heavier type that stays legible when a
    # multi-panel figure is printed at A0 and viewed from a distance, plus
    # high raster fidelity for the PNG export.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.dpi": 150,
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2937",
            "axes.linewidth": 1.1,
            "axes.labelcolor": "#111827",
            "axes.labelweight": "medium",
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.titlepad": 11,
            "axes.labelsize": 12,
            "axes.labelpad": 5,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10.5,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.6,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 600,
            "xtick.color": "#1f2937",
            "ytick.color": "#1f2937",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    return pd.read_csv(path, sep="\t")


def model_label(model_key: str) -> str:
    return MODEL_LABELS.get(str(model_key), str(model_key).replace("-", " ").title())


def study_label(study: str) -> str:
    """Human-readable cohort name (underscores read as spaces)."""
    return str(study).replace("_", " ")


def study_type(study: str) -> str:
    """Coarse disease-category group for a cohort (for the heatmap colour strip).

    Maps a cohort to the same coarse category used by
    ``plot_disease_category_stats`` (e.g. ``Cancer (CRC)``, ``IBD``,
    ``Metabolic``). Cohorts without an assignment fall back to ``Unknown``.
    """
    specific = STUDY_DISEASE.get(str(study))
    if specific is None:
        return "Unknown"
    return CATEGORY_GROUP.get(specific, "Other")


def study_type_colors() -> dict[str, str]:
    """Colour map for the study-type strip, ordered for a stable legend."""
    colors = dict(CATEGORY_COLORS)
    colors.setdefault("Unknown", UNKNOWN_TYPE_COLOR)
    return colors


def family_for_model(model_key: str) -> str:
    key = str(model_key).lower()
    if key.startswith("xgb"):
        return "baseline"
    # Compound prefixes must be matched before their substrings: "phylodiva"
    # contains "diva", and "taxi" contains "tax".
    if "phylodiva" in key:
        return "fusion"
    if "taxi" in key:
        return "taxonomy"
    if "diva" in key:
        return "diva"
    if "tree" in key or "treenb" in key:
        return "tree"
    if "philr" in key:
        return "philr"
    if "hyp" in key:
        return "hyperbolic"
    if "tax" in key:
        return "taxonomy"
    if "graph" in key:
        return "graph"
    if "dsvae" in key:
        return "dsvae"
    if "fuse" in key:
        return "fusion"
    return "vanilla"


def model_color(model_key: str) -> str:
    return FAMILY_COLORS[family_for_model(model_key)]


def clean_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def summarize_models(
    df: pd.DataFrame,
    model_col: str,
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for model, group in df.groupby(model_col, sort=False):
        row: dict[str, float | str | int] = {
            "model_key": model,
            "model_label": model_label(str(model)),
            "n_rows": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_sem"] = (
                float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    if "balanced_accuracy_mean" in summary.columns:
        summary = summary.sort_values("balanced_accuracy_mean", ascending=False)
    return summary.reset_index(drop=True)


def add_panel_label(ax, label: str) -> None:
    # Anchor the panel letter to the figure-relative top-left corner of the
    # axes and lift it clearly above the (centred) title line so the two
    # never collide, regardless of how wide the title is.
    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-42, 26),
        textcoords="offset points",
        fontsize=17,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#0f172a",
        annotation_clip=False,
    )


def style_axis(ax, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.9)


def add_family_legend(fig, families: Iterable[str]) -> None:
    """Place a shared legend that decodes the model-family colour scheme.

    The legend gets its own reserved band at the bottom of the figure: the
    constrained-layout engine packs every axis (and its rotated tick labels)
    above that band, so the legend can never collide with panel content.
    """
    from matplotlib.patches import Patch

    seen: list[str] = []
    for fam in families:
        if fam in FAMILY_COLORS and fam not in seen:
            seen.append(fam)
    handles = [
        Patch(facecolor=FAMILY_COLORS[fam], edgecolor="white", label=FAMILY_LABELS.get(fam, fam.title()))
        for fam in seen
    ]
    ncol = min(len(handles), 5)
    nrows = max(1, math.ceil(len(handles) / ncol))
    # Reserve vertical room for the legend (title line + one line per row).
    # Size the reservation in inches so the absolute gap between the legend and
    # the axis labels stays consistent whether this is the tall multipanel or a
    # short standalone single-panel figure (a fixed fraction would crowd the
    # axis label on short figures).
    fig_h = float(fig.get_size_inches()[1])
    band = min(0.32, (0.5 + 0.55 * nrows) / fig_h)
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(rect=(0.0, band, 1.0, 1.0 - band))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        fontsize=11,
        title="Model family",
        title_fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.004),
        handlelength=1.3,
        columnspacing=1.8,
        handletextpad=0.6,
    )


def place_repelled_labels(
    ax,
    xs,
    ys,
    labels,
    point_sizes=None,
    fontsize: float = 8.0,
    n_iter: int = 900,
    pad: float = 4.0,
) -> None:
    """Label every scatter point with non-overlapping text + leader lines.

    A dependency-free re-implementation of the adjustText force model: labels
    repel one another and the data points in pixel space, then settle into
    data coordinates (so they ride with the axis through constrained-layout
    and the tight-bbox crop). Keeps a crowded scatter human-readable.
    """
    from matplotlib.transforms import IdentityTransform

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    # Marker radii in pixels (scatter ``s`` is a point-area), so labels can be
    # pushed clear of the bubble, not merely its centre.
    if point_sizes is not None:
        radii = np.sqrt(np.asarray(point_sizes, dtype=float) / math.pi) * fig.dpi / 72.0
    else:
        radii = np.zeros(len(xs))
    # Seed each label on a distinct golden-angle direction so that points
    # sharing a location (or tightly clustered) start apart — identical seeds
    # would move in lockstep under the symmetric forces and never separate.
    ang = np.arange(len(xs)) * 2.399963229728653
    lab = pts + 16.0 * np.column_stack([np.cos(ang), np.sin(ang)])

    tmp = [
        ax.text(0, 0, s, fontsize=fontsize, fontweight="medium", transform=IdentityTransform())
        for s in labels
    ]
    w = np.empty(len(tmp))
    h = np.empty(len(tmp))
    for k, t in enumerate(tmp):
        bb = t.get_window_extent(renderer=renderer)
        w[k] = bb.width + pad * 2
        h[k] = bb.height + pad * 2
    for t in tmp:
        t.remove()

    n = len(labels)
    for _ in range(n_iter):
        moved = False
        for i in range(n):
            fx = fy = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = lab[i, 0] - lab[j, 0]
                dy = lab[i, 1] - lab[j, 1]
                ox = (w[i] + w[j]) / 2 - abs(dx)
                oy = (h[i] + h[j]) / 2 - abs(dy)
                if ox > 0 and oy > 0:
                    if ox <= oy:
                        fx += math.copysign(ox, dx if dx != 0 else 1.0)
                    else:
                        fy += math.copysign(oy, dy if dy != 0 else 1.0)
            for j in range(n):
                dx = lab[i, 0] - pts[j, 0]
                dy = lab[i, 1] - pts[j, 1]
                clear = radii[j] + 4.0
                ox = w[i] / 2 + clear - abs(dx)
                oy = h[i] / 2 + clear - abs(dy)
                if ox > 0 and oy > 0:
                    if ox <= oy:
                        fx += math.copysign(ox, dx if dx != 0 else 1.0)
                    else:
                        fy += math.copysign(oy, dy if dy != 0 else 1.0)
            if abs(fx) > 0.01 or abs(fy) > 0.01:
                lab[i, 0] += 0.5 * fx
                lab[i, 1] += 0.5 * fy
                moved = True
        if not moved:
            break

    ab = ax.get_window_extent(renderer=renderer)
    lab[:, 0] = np.clip(lab[:, 0], ab.x0 + w / 2, ab.x1 - w / 2)
    lab[:, 1] = np.clip(lab[:, 1], ab.y0 + h / 2, ab.y1 - h / 2)
    lab_data = ax.transData.inverted().transform(lab)
    for (lx, ly), px, py, s in zip(lab_data, xs, ys, labels):
        ax.annotate(
            s,
            xy=(px, py),
            xytext=(lx, ly),
            textcoords="data",
            fontsize=fontsize,
            fontweight="medium",
            color="#1f2937",
            ha="center",
            va="center",
            zorder=5,
            arrowprops=dict(arrowstyle="-", color="#aeb4bd", lw=0.6, shrinkA=0, shrinkB=2),
        )


def save_figure(fig, outdir: Path, stem: str, formats: Iterable[str]) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        # 600 dpi raster for crisp poster printing; vector formats keep text
        # selectable and infinitely scalable.
        dpi = 600 if fmt == "png" else 1200
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
        paths.append(path)
    return paths


def plot_horizontal_metric(
    ax,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    xlim: tuple[float, float],
) -> None:
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    data = summary.dropna(subset=[mean_col]).sort_values(mean_col, ascending=True)
    y = np.arange(len(data))
    colors = [model_color(k) for k in data["model_key"]]
    errs = data[sem_col].fillna(0.0).to_numpy() * 1.96 if sem_col in data else None
    bars = ax.barh(
        y,
        data[mean_col],
        color=colors,
        alpha=0.95,
        height=0.72,
        edgecolor="white",
        linewidth=0.6,
    )
    if errs is not None and np.nanmax(errs) > 0:
        ax.errorbar(
            data[mean_col],
            y,
            xerr=errs,
            fmt="none",
            ecolor="#1f2937",
            elinewidth=1.0,
            capsize=2.8,
            capthick=1.0,
        )
    # Value labels just outside each bar so exact numbers read at a glance.
    span = xlim[1] - xlim[0]
    err_arr = errs if errs is not None else np.zeros(len(data))
    for bar, val, err in zip(bars, data[mean_col].to_numpy(), err_arr):
        ax.text(
            min(val + (err if np.isfinite(err) else 0.0) + span * 0.012, xlim[1] - span * 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#374151",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data["model_label"], fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.margins(y=0.01)
    style_axis(ax, "x")


def plot_heatmap(
    ax,
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    annotate: bool = False,
    xtick_fontsize: float = 9.0,
    ytick_fontsize: float = 8.5,
    row_categories: list[str] | None = None,
    category_colors: dict[str, str] | None = None,
    type_legend: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    arr = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    image = ax.imshow(
        masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [model_label(c) for c in matrix.columns], rotation=45, ha="right", fontsize=xtick_fontsize
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    yticklabels = [study_label(s) for s in matrix.index]
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Optional study-type colour strip down the left margin. The cohort names
    # move from the heatmap onto the strip, so the left-to-right reading order
    # becomes [cohort labels] [type strip] [heatmap] [colourbar].
    if row_categories is not None:
        from matplotlib.colors import ListedColormap

        cats = [str(c) for c in row_categories]
        if category_colors is None:
            category_colors = {}
        present = list(dict.fromkeys(cats))
        # Index categories by the colour-map ordering when available so the
        # strip and its legend share a stable, meaningful order.
        ordered = [c for c in category_colors if c in present]
        ordered += [c for c in present if c not in ordered]
        idx_of = {c: i for i, c in enumerate(ordered)}
        strip = np.array([[idx_of[c]] for c in cats], dtype=float)
        strip_cmap = ListedColormap(
            [category_colors.get(c, UNKNOWN_TYPE_COLOR) for c in ordered]
        )
        strip_ax = ax.inset_axes([-0.05, 0.0, 0.024, 1.0])
        strip_ax.imshow(
            strip, aspect="auto", cmap=strip_cmap,
            vmin=-0.5, vmax=len(ordered) - 0.5, interpolation="nearest",
        )
        strip_ax.set_xticks([])
        strip_ax.set_yticks(np.arange(len(cats)))
        strip_ax.set_yticklabels(yticklabels, fontsize=ytick_fontsize)
        strip_ax.tick_params(length=0)
        for spine in strip_ax.spines.values():
            spine.set_visible(False)
        ax.set_yticklabels([])  # cohort labels now live on the strip
        if type_legend:
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=category_colors.get(c, UNKNOWN_TYPE_COLOR),
                      edgecolor="white", label=c)
                for c in ordered
            ]
            ax.legend(
                handles=handles,
                title="Study type",
                loc="upper left",
                bbox_to_anchor=(1.28, 1.0),
                fontsize=8.5,
                title_fontsize=9.5,
                frameon=False,
                handlelength=1.1,
                handletextpad=0.6,
                borderaxespad=0.0,
            )
    else:
        ax.set_yticklabels(yticklabels, fontsize=ytick_fontsize)

    if annotate:
        lo = vmin if vmin is not None else float(np.nanmin(arr))
        hi = vmax if vmax is not None else float(np.nanmax(arr))
        mid = (lo + hi) / 2.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if val < mid else "#111827",
                )
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9.5)
    cbar.outline.set_visible(False)


def plot_single_study(meta: pd.DataFrame, outdir: Path, formats: Iterable[str]) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    metrics = ["balanced_accuracy", "f1_macro", "auroc", "rmse", "mae"]
    meta = clean_numeric(meta, metrics + [f"{m}_std" for m in metrics])
    summary = summarize_models(meta, "model_key", metrics)
    summary.to_csv(outdir / "single_study_model_summary.tsv", sep="\t", index=False)

    order_by_auroc = (
        summary.sort_values("auroc_mean", ascending=False)["model_key"].astype(str).tolist()
    )
    study_order = (
        meta.groupby("study")["auroc"].mean().sort_values(ascending=False).index.tolist()
    )
    heat = (
        meta.pivot_table(index="study", columns="model_key", values="auroc", aggfunc="mean")
        .reindex(index=study_order, columns=order_by_auroc)
    )
    # Disease-category of each cohort, aligned to the heatmap rows, drives the
    # colour strip added to the left of the AUROC heatmap.
    heat_row_types = [study_type(s) for s in heat.index]
    type_colors = study_type_colors()

    winners = (
        meta.loc[meta.groupby("study")["auroc"].idxmax(), "model_key"]
        .value_counts()
        .rename_axis("model_key")
        .reset_index(name="n_best_studies")
    )
    winners["model_label"] = winners["model_key"].map(model_label)

    ranked = meta.copy()
    for metric, ascending in [
        ("auroc", False),
        ("balanced_accuracy", False),
        ("f1_macro", False),
        ("rmse", True),
        ("mae", True),
    ]:
        ranked[f"{metric}_rank"] = ranked.groupby("study")[metric].rank(
            method="average", ascending=ascending
        )
    rank_cols = [f"{m}_rank" for m in ["auroc", "balanced_accuracy", "f1_macro", "rmse", "mae"]]
    rank_summary = (
        ranked.groupby("model_key")[rank_cols]
        .mean()
        .mean(axis=1)
        .sort_values(ascending=True)
        .reset_index(name="mean_multi_metric_rank")
    )
    rank_summary["model_label"] = rank_summary["model_key"].map(model_label)

    # ----- per-panel drawing functions (shared by multipanel + standalone) ---
    def draw_auroc(ax):
        plot_horizontal_metric(
            ax, summary, "auroc", "AUROC ranking across 42 studies",
            (0.45, min(1.0, max(0.92, float(summary["auroc_mean"].max()) + 0.05))),
        )

    def draw_balanced_accuracy(ax):
        plot_horizontal_metric(
            ax, summary, "balanced_accuracy", "Balanced accuracy ranking",
            (0.35, min(1.0, max(0.82, float(summary["balanced_accuracy_mean"].max()) + 0.05))),
        )

    def draw_rmse(ax):
        plot_horizontal_metric(
            ax, summary, "rmse", "Reconstruction error (lower better)",
            (0.0, float(summary["rmse_mean"].max()) + 0.04),
        )

    def draw_winners(ax):
        top_winners = winners.sort_values("n_best_studies", ascending=True)
        bars = ax.barh(
            np.arange(len(top_winners)),
            top_winners["n_best_studies"],
            color=[model_color(k) for k in top_winners["model_key"]],
            height=0.72, edgecolor="white", linewidth=0.6,
        )
        d_max = float(top_winners["n_best_studies"].max())
        for bar, val in zip(bars, top_winners["n_best_studies"].to_numpy()):
            if val > 0:
                ax.text(
                    bar.get_width() + d_max * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(val)}", va="center", ha="left", fontsize=9, color="#374151",
                )
        ax.set_xlim(0, d_max * 1.12)
        ax.set_yticks(np.arange(len(top_winners)))
        ax.set_yticklabels(top_winners["model_label"], fontsize=10)
        ax.set_xlabel("Studies won by AUROC")
        ax.set_title("Best model frequency")
        style_axis(ax, "x")

    def draw_scatter(ax):
        scatter = summary.dropna(subset=["rmse_mean", "auroc_mean"]).copy()
        sizes = 90 + 16 * scatter["n_rows"].astype(float)
        ax.scatter(
            scatter["rmse_mean"], scatter["auroc_mean"], s=sizes,
            c=[model_color(k) for k in scatter["model_key"]],
            alpha=0.88, edgecolor="white", linewidth=1.1, zorder=3,
        )
        # Faint trend line + Pearson r summarises the accuracy/reconstruction
        # trade-off without relying on every point being individually readable.
        sx = scatter["rmse_mean"].to_numpy(dtype=float)
        sy = scatter["auroc_mean"].to_numpy(dtype=float)
        if len(sx) >= 3:
            slope, intercept = np.polyfit(sx, sy, 1)
            xline = np.linspace(sx.min(), sx.max(), 50)
            ax.plot(xline, slope * xline + intercept, ls="--", lw=1.2, color="#9ca3af", zorder=2)
            r = float(np.corrcoef(sx, sy)[0, 1])
            ax.annotate(
                f"Pearson r = {r:.2f}", xy=(0.03, 0.04), xycoords="axes fraction",
                fontsize=9.5, color="#4b5563", ha="left", va="bottom",
            )
        ax.set_xlabel("Mean RMSE (lower is better)")
        ax.set_ylabel("Mean AUROC (higher is better)")
        ax.set_title("Accuracy versus reconstruction")
        ax.margins(0.18)
        style_axis(ax, "both")
        # Force-directed labels so every model is identifiable without collisions.
        place_repelled_labels(
            ax, sx, sy,
            [model_label(k) for k in scatter["model_key"]],
            point_sizes=sizes.to_numpy(dtype=float), fontsize=8.0,
        )

    def draw_heatmap(ax):
        plot_heatmap(
            ax, heat, "AUROC by study and model", "AUROC",
            vmin=0.45, vmax=1.0, cmap="viridis",
            xtick_fontsize=8.5, ytick_fontsize=7.5,
            row_categories=heat_row_types, category_colors=type_colors,
            type_legend=True,
        )

    def draw_rank(ax):
        rank_plot = rank_summary.sort_values("mean_multi_metric_rank", ascending=True)
        bars = ax.bar(
            np.arange(len(rank_plot)), rank_plot["mean_multi_metric_rank"],
            color=[model_color(k) for k in rank_plot["model_key"]],
            width=0.78, edgecolor="white", linewidth=0.6,
        )
        for bar, val in zip(bars, rank_plot["mean_multi_metric_rank"].to_numpy()):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color="#374151",
            )
        ax.set_xticks(np.arange(len(rank_plot)))
        ax.set_xticklabels(rank_plot["model_label"], rotation=35, ha="right", fontsize=10)
        ax.set_ylabel("Mean rank (AUROC, BA, F1, RMSE, MAE)")
        ax.set_title("Overall multi-metric ordering (lower rank is better)")
        ax.margins(x=0.01)
        # Headroom so the tallest bar's value label clears the top spine.
        ax.set_ylim(0, float(rank_plot["mean_multi_metric_rank"].max()) * 1.12)
        style_axis(ax, "y")

    # (letter, filename slug, standalone figsize, draw fn, show family legend)
    panels = [
        ("A", "auroc_ranking", (7.5, 6.5), draw_auroc, True),
        ("B", "balanced_accuracy_ranking", (7.5, 6.5), draw_balanced_accuracy, True),
        ("C", "reconstruction_error", (7.5, 6.5), draw_rmse, True),
        ("D", "best_model_frequency", (7.5, 5.0), draw_winners, True),
        ("E", "accuracy_vs_reconstruction", (8.0, 6.5), draw_scatter, True),
        ("F", "auroc_heatmap", (9.5, 12.0), draw_heatmap, False),
        ("G", "multi_metric_ranking", (13.0, 5.0), draw_rank, False),
    ]
    families_present = [family_for_model(k) for k in summary["model_key"]]

    # ----- multipanel figure -------------------------------------------------
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    grid = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.05, 1.25])
    axes_map = {
        "A": fig.add_subplot(grid[0, 0]),
        "B": fig.add_subplot(grid[0, 1]),
        "C": fig.add_subplot(grid[0, 2]),
        "D": fig.add_subplot(grid[1, 0]),
        "E": fig.add_subplot(grid[1, 1]),
        "F": fig.add_subplot(grid[1:, 2]),
        "G": fig.add_subplot(grid[2, 0:2]),
    }
    for letter, _slug, _figsize, draw, _legend in panels:
        ax = axes_map[letter]
        draw(ax)
        add_panel_label(ax, letter)
    add_family_legend(fig, families_present)
    paths = save_figure(fig, outdir, "single_study_results_multipanel", formats)
    plt.close(fig)

    # ----- standalone single-panel figures (no panel letters) ----------------
    panel_dir = outdir / "single_study_panels"
    for letter, slug, figsize, draw, show_family in panels:
        sfig = plt.figure(figsize=figsize, constrained_layout=True)
        sax = sfig.add_subplot(1, 1, 1)
        draw(sax)
        if show_family:
            add_family_legend(sfig, families_present)
        paths += save_figure(sfig, panel_dir, f"single_study_panel_{letter.lower()}_{slug}", formats)
        plt.close(sfig)

    return paths


def plot_loso(loso: pd.DataFrame, meta: pd.DataFrame, outdir: Path, formats: Iterable[str]) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    metrics = ["balanced_accuracy", "f1_macro", "auroc"]
    loso = clean_numeric(loso, metrics + [f"{m}_std" for m in metrics] + ["n_eval_samples"])
    meta = clean_numeric(meta, metrics)
    loso_summary = summarize_models(loso, "model", ["balanced_accuracy", "f1_macro"])
    loso_summary.to_csv(outdir / "strict_loso_model_summary.tsv", sep="\t", index=False)

    order_by_ba = (
        loso_summary.sort_values("balanced_accuracy_mean", ascending=False)["model_key"]
        .astype(str)
        .tolist()
    )
    study_order = (
        loso.groupby("held_out_study")["balanced_accuracy"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    ba_heat = (
        loso.pivot_table(
            index="held_out_study",
            columns="model",
            values="balanced_accuracy",
            aggfunc="mean",
        )
        .reindex(index=study_order, columns=order_by_ba)
    )
    f1_heat = (
        loso.pivot_table(index="held_out_study", columns="model", values="f1_macro", aggfunc="mean")
        .reindex(index=study_order, columns=order_by_ba)
    )

    winners = (
        loso.loc[loso.groupby("held_out_study")["balanced_accuracy"].idxmax(), "model"]
        .value_counts()
        .rename_axis("model_key")
        .reset_index(name="n_best_folds")
    )
    winners = (
        pd.DataFrame({"model_key": order_by_ba})
        .merge(winners, on="model_key", how="left")
        .fillna({"n_best_folds": 0})
    )
    winners["n_best_folds"] = winners["n_best_folds"].astype(int)
    winners["model_label"] = winners["model_key"].map(model_label)

    common_models = sorted(set(loso["model"].unique()) & set(meta["model_key"].unique()))
    common_studies = sorted(set(loso["held_out_study"].unique()) & set(meta["study"].unique()))
    gap_rows = []
    for model in common_models:
        for study in common_studies:
            single_val = meta.loc[
                (meta["model_key"] == model) & (meta["study"] == study),
                "balanced_accuracy",
            ]
            loso_val = loso.loc[
                (loso["model"] == model) & (loso["held_out_study"] == study),
                "balanced_accuracy",
            ]
            if len(single_val) and len(loso_val):
                gap_rows.append(
                    {
                        "model_key": model,
                        "study": study,
                        "single_balanced_accuracy": float(single_val.iloc[0]),
                        "loso_balanced_accuracy": float(loso_val.iloc[0]),
                        "gap": float(single_val.iloc[0] - loso_val.iloc[0]),
                    }
                )
    gap = pd.DataFrame(gap_rows)
    if len(gap):
        gap.to_csv(outdir / "single_vs_strict_loso_gaps.tsv", sep="\t", index=False)

    study_difficulty = (
        loso.groupby("held_out_study")
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            f1_macro=("f1_macro", "mean"),
            n_eval_samples=("n_eval_samples", "first"),
        )
        .sort_values("balanced_accuracy", ascending=True)
    )

    # ----- per-panel drawing functions (shared by multipanel + standalone) ---
    def draw_ba_ranking(ax):
        plot_horizontal_metric(
            ax, loso_summary, "balanced_accuracy", "Strict LOSO balanced accuracy",
            (0.25, min(1.0, max(0.8, float(loso_summary["balanced_accuracy_mean"].max()) + 0.06))),
        )

    def draw_f1_ranking(ax):
        plot_horizontal_metric(
            ax, loso_summary, "f1_macro", "Strict LOSO macro F1",
            (0.15, min(0.75, max(0.48, float(loso_summary["f1_macro_mean"].max()) + 0.08))),
        )

    def draw_best_model(ax):
        win_plot = winners.sort_values(["n_best_folds", "model_key"], ascending=[True, True])
        bars = ax.barh(
            np.arange(len(win_plot)), win_plot["n_best_folds"],
            color=[model_color(k) for k in win_plot["model_key"]],
            height=0.72, edgecolor="white", linewidth=0.6,
        )
        c_max = max(1.0, float(win_plot["n_best_folds"].max()))
        for bar, val in zip(bars, win_plot["n_best_folds"].to_numpy()):
            if val > 0:
                ax.text(
                    bar.get_width() + c_max * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(val)}", va="center", ha="left", fontsize=9.5, color="#374151",
                )
        ax.set_xlim(0, c_max * 1.12)
        ax.set_yticks(np.arange(len(win_plot)))
        ax.set_yticklabels(win_plot["model_label"], fontsize=10)
        ax.set_xlabel("Held-out studies won")
        ax.set_title("Best-model frequency")
        style_axis(ax, "x")

    def draw_ba_heatmap(ax):
        plot_heatmap(
            ax, ba_heat, "Balanced accuracy by held-out study", "Balanced accuracy",
            vmin=0.25, vmax=1.0, cmap="viridis", annotate=True,
            xtick_fontsize=10, ytick_fontsize=9.5,
        )

    def draw_f1_heatmap(ax):
        plot_heatmap(
            ax, f1_heat, "Macro F1 by held-out study", "Macro F1",
            vmin=float(np.nanmin(f1_heat.to_numpy())),
            vmax=float(np.nanmax(f1_heat.to_numpy())),
            cmap="cividis", annotate=True,
            xtick_fontsize=9.5, ytick_fontsize=9.5,
        )

    def draw_difficulty(ax):
        y = np.arange(len(study_difficulty))
        ax.barh(y, study_difficulty["balanced_accuracy"], color="#2563eb", alpha=0.9,
                height=0.72, edgecolor="white", linewidth=0.6)
        for yi, val in zip(y, study_difficulty["balanced_accuracy"].to_numpy()):
            ax.text(val + 0.012, yi, f"{val:.2f}", va="center", ha="left", fontsize=9, color="#374151")
        ax.set_yticks(y)
        ax.set_yticklabels([study_label(s) for s in study_difficulty.index], fontsize=10)
        ax.set_xlim(0.25, 1.02)
        ax.set_xlabel("Mean balanced accuracy")
        ax.set_title("Held-out cohort difficulty")
        style_axis(ax, "x")

    def draw_cohort_size(ax):
        ax.bar(
            np.arange(len(study_difficulty)), study_difficulty["n_eval_samples"],
            color="#64748b", width=0.76, edgecolor="white", linewidth=0.6,
        )
        ax.set_xticks(np.arange(len(study_difficulty)))
        ax.set_xticklabels(
            [study_label(s) for s in study_difficulty.index],
            rotation=45, ha="right", fontsize=9.5,
        )
        ax.set_ylabel("Held-out samples")
        ax.set_title("Evaluation cohort size")
        style_axis(ax, "y")

    def draw_transfer_gap(ax):
        if len(gap):
            gap_plot = (
                gap.groupby("model_key")
                .agg(
                    single_balanced_accuracy=("single_balanced_accuracy", "mean"),
                    loso_balanced_accuracy=("loso_balanced_accuracy", "mean"),
                    gap=("gap", "mean"),
                )
                .sort_values("gap", ascending=False)
            )
            x = np.arange(len(gap_plot))
            width = 0.36
            ax.bar(
                x - width / 2, gap_plot["single_balanced_accuracy"], width=width,
                color="#93c5fd", edgecolor="white", linewidth=0.6, label="Within-study",
            )
            ax.bar(
                x + width / 2, gap_plot["loso_balanced_accuracy"], width=width,
                color="#1d4ed8", edgecolor="white", linewidth=0.6, label="Strict LOSO",
            )
            ymax = min(
                1.0,
                max(
                    0.68,
                    float(
                        gap_plot[["single_balanced_accuracy", "loso_balanced_accuracy"]]
                        .max()
                        .max()
                    )
                    + 0.14,
                ),
            )
            for idx, (_, row) in enumerate(gap_plot.iterrows()):
                ax.text(
                    idx,
                    max(row["single_balanced_accuracy"], row["loso_balanced_accuracy"]) + 0.02,
                    f"Δ{row['gap']:+.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="medium", color="#374151",
                )
            ax.set_xticks(x)
            ax.set_xticklabels([model_label(k) for k in gap_plot.index], rotation=35, ha="right", fontsize=9.5)
            ax.set_ylim(0.25, ymax)
            ax.set_ylabel("Balanced accuracy")
            ax.set_title("Within-study → LOSO transfer gap")
            ax.legend(
                loc="upper center", ncol=2, fontsize=9.5, frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.9,
                columnspacing=1.4, handlelength=1.2,
            )
            style_axis(ax, "y")
        else:
            ax.axis("off")
            ax.text(
                0.5, 0.5, "No shared model/study pairs for gap analysis",
                ha="center", va="center", fontsize=12,
            )

    # (letter, filename slug, standalone figsize, draw fn, show family legend)
    panels = [
        ("A", "ba_ranking", (7.5, 5.5), draw_ba_ranking, True),
        ("B", "f1_ranking", (7.5, 5.5), draw_f1_ranking, True),
        ("C", "best_model_frequency", (7.5, 5.0), draw_best_model, True),
        ("D", "ba_heatmap", (13.0, 10.0), draw_ba_heatmap, False),
        ("E", "f1_heatmap", (8.5, 10.0), draw_f1_heatmap, False),
        ("F", "cohort_difficulty", (7.5, 9.0), draw_difficulty, False),
        ("G", "cohort_size", (11.0, 5.0), draw_cohort_size, False),
        ("H", "transfer_gap", (9.0, 5.5), draw_transfer_gap, False),
    ]
    families_present = [family_for_model(k) for k in loso_summary["model_key"]]

    # ----- multipanel figure -------------------------------------------------
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.1])
    axes_map = {
        "A": fig.add_subplot(grid[0, 0]),
        "B": fig.add_subplot(grid[0, 1]),
        "C": fig.add_subplot(grid[0, 2]),
        "D": fig.add_subplot(grid[1, 0:2]),
        "E": fig.add_subplot(grid[1, 2]),
        "F": fig.add_subplot(grid[2, 0]),
        "G": fig.add_subplot(grid[2, 1]),
        "H": fig.add_subplot(grid[2, 2]),
    }
    for letter, _slug, _figsize, draw, _legend in panels:
        ax = axes_map[letter]
        draw(ax)
        add_panel_label(ax, letter)
    add_family_legend(fig, families_present)
    paths = save_figure(fig, outdir, "strict_loso_results_multipanel", formats)
    plt.close(fig)

    # ----- standalone single-panel figures (no panel letters) ----------------
    panel_dir = outdir / "strict_loso_panels"
    for letter, slug, figsize, draw, show_family in panels:
        sfig = plt.figure(figsize=figsize, constrained_layout=True)
        sax = sfig.add_subplot(1, 1, 1)
        draw(sax)
        if show_family:
            add_family_legend(sfig, families_present)
        paths += save_figure(sfig, panel_dir, f"strict_loso_panel_{letter.lower()}_{slug}", formats)
        plt.close(sfig)

    return paths


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.outdir.mkdir(parents=True, exist_ok=True)
    meta = read_table(args.meta)
    loso = read_table(args.loso)

    paths = []
    paths.extend(plot_single_study(meta, args.outdir, args.formats))
    paths.extend(plot_loso(loso, meta, args.outdir, args.formats))

    print("Wrote result figures:")
    for path in paths:
        print(f"  {path}")
    print(f"Wrote aggregate tables to {args.outdir}")


if __name__ == "__main__":
    main()
