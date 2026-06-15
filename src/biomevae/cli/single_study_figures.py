"""Generate publication-quality figures for a single microbiome study run.

Expects a biomevae results directory with one sub-folder per trained model
(as produced by ``hpc/single_study_pipeline.sh``) and the three pre-extracted
data files (``sgb_table.tsv``, ``phyla.tsv``, ``sample_metadata.tsv``)
produced by the extract-microbiome-data package
(https://github.com/rvazdev-ex/extract-microbiome-data).

Produces a multi-panel figure suite suitable for a methods paper:

1. **Latent space ordination** – PCA and UMAP of VAE embeddings coloured by
   the chosen metadata label, one panel per model.
2. **Classification performance** – Grouped bar chart of balanced accuracy,
   F1-macro, and AUROC across models and classifiers.
3. **Confusion matrices** – Heatmaps for each model's best classifier.
4. **Reconstruction quality** – Per-model R², MAE, and RMSE bar charts.
5. **Training curves** – Loss trajectories for all trained models.
6. **Feature importance** – Top-20 SHAP features for the best-performing model
   (if available).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_embeddings(path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df, list(df.index)


def _load_metadata(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def _load_training_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _discover_models(results_dir: str) -> Dict[str, Path]:
    """Discover trained model directories under results_dir."""
    models = {}
    root = Path(results_dir)
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            models[d.name] = d
    return models


_MODEL_DISPLAY_NAMES = {
    "beta-vae": r"$\beta$-VAE",
    "vanilla-vae": "Vanilla VAE",
    "hyp-vae": "Hyperbolic VAE",
    "tax-vae": "Tax-aware VAE",
    "hyp-tax-vae": "Hyp+Tax VAE",
    "graph-vae": "Graph VAE",
    "treeprior-vae": "TreePrior VAE",
    "fuse-vae": "PhyloFusion VAE",
    "tree-dtm-vae": "TreeDTM-VAE",
    "philrvae": "PhILR-VAE",
    "hyperbolic-philrvae": "Hyp-PhILR-VAE",
    "hyp-philrvae": "Hyp-PhILR-NB VAE",
    "hyp-philr-zinb": "Hyp-PhILR-ZINB VAE",
    "xgboost-baseline": "XGBoost (SGB)",
}


def _display_name(model_key: str) -> str:
    return _MODEL_DISPLAY_NAMES.get(model_key, model_key)


def _load_latent_dim(model_dir: Path) -> Optional[int]:
    """Read optimal latent dimension from a model's config.json."""
    config_path = model_dir / "config.json"
    if config_path.exists():
        cfg = _load_json(str(config_path))
        d = cfg.get("latent_dim")
        if d is not None:
            return int(d)
    return None


# ---------------------------------------------------------------------------
# Figure 1: Latent space ordination (PCA + UMAP) coloured by disease
# ---------------------------------------------------------------------------

def _build_colour_map(unique_labels: List[str]) -> Dict[str, str]:
    """Build a colour map for a set of class labels.

    Uses well-known colours for common disease labels, then falls back to
    the matplotlib ``tab10`` palette for everything else.
    """
    import matplotlib.cm as cm

    known = {
        "CRC": "#d62728",
        "healthy": "#2ca02c",
        "control": "#2ca02c",
        "adenoma": "#ff7f0e",
        "IBD": "#9467bd",
        "CDI": "#8c564b",
        "T2D": "#e377c2",
    }
    cmap = {}
    tab10_idx = 0
    tab10 = cm.get_cmap("tab10")
    for lab in sorted(unique_labels):
        if lab in known:
            cmap[lab] = known[lab]
        else:
            # pick next tab10 colour not already used
            while True:
                hex_col = "#{:02x}{:02x}{:02x}".format(
                    *[int(c * 255) for c in tab10(tab10_idx % 10)[:3]]
                )
                tab10_idx += 1
                if hex_col not in cmap.values():
                    break
            cmap[lab] = hex_col
    return cmap


def fig_latent_ordination(
    models: Dict[str, Path],
    metadata_path: str,
    outdir: str,
    label_col: str = "disease",
    study_name: str = "Study",
) -> Optional[str]:
    """PCA + UMAP of embeddings coloured by metadata class for each model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    try:
        from umap import UMAP
        has_umap = True
    except ImportError:
        has_umap = False

    meta = _load_metadata(metadata_path)
    if "sample_id" in meta.columns:
        meta = meta.set_index("sample_id")
    if label_col not in meta.columns:
        print(f"  WARNING: no '{label_col}' column in metadata; skipping ordination.")
        return None

    embed_models = {}
    for name, mdir in models.items():
        embed_path = mdir / "embed" / "embeddings.tsv"
        if not embed_path.exists():
            embed_path = mdir / "test" / "embeddings.tsv"
        if not embed_path.exists():
            embed_path = mdir / "embeddings.tsv"
        if embed_path.exists():
            embed_models[name] = embed_path

    if not embed_models:
        print("  No embeddings found; skipping ordination figure.")
        return None

    # Collect all unique labels across models for a consistent colour map
    all_labels = set()
    for epath in embed_models.values():
        _, samples = _load_embeddings(str(epath))
        labs = meta.reindex(samples)[label_col].fillna("unknown")
        all_labels.update(str(l).strip() for l in labs)
    colour_map = _build_colour_map(sorted(all_labels))

    n_models = len(embed_models)
    n_cols = min(n_models, 4)
    n_rows_per_method = (n_models + n_cols - 1) // n_cols
    n_method_blocks = 2 if has_umap else 1  # PCA + optionally UMAP
    n_rows = n_rows_per_method * n_method_blocks

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (name, epath) in enumerate(embed_models.items()):
        emb_df, samples = _load_embeddings(str(epath))
        labels = meta.reindex(samples)[label_col].fillna("unknown")
        labels = labels.apply(lambda x: str(x).strip())
        latent_dim = _load_latent_dim(models[name])

        # PCA
        pca = PCA(n_components=2, random_state=42)
        Z_pca = pca.fit_transform(emb_df.values)

        row_pca = idx // n_cols
        col = idx % n_cols
        ax = axes[row_pca, col]
        for lab in sorted(colour_map):
            mask = labels == lab
            if mask.sum() > 0:
                ax.scatter(
                    Z_pca[mask, 0], Z_pca[mask, 1],
                    c=colour_map[lab], label=lab, s=18, alpha=0.7, edgecolors="none",
                )
        title = f"{_display_name(name)} – PCA"
        if latent_dim is not None:
            title += f"  ($d$={latent_dim})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="best", framealpha=0.6)

        # UMAP
        if has_umap:
            Z_umap = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(emb_df.values)
            row_umap = n_rows_per_method + idx // n_cols
            ax2 = axes[row_umap, col]
            for lab in sorted(colour_map):
                mask = labels == lab
                if mask.sum() > 0:
                    ax2.scatter(
                        Z_umap[mask, 0], Z_umap[mask, 1],
                        c=colour_map[lab], label=lab, s=18, alpha=0.7, edgecolors="none",
                    )
            title_umap = f"{_display_name(name)} – UMAP"
            if latent_dim is not None:
                title_umap += f"  ($d$={latent_dim})"
            ax2.set_title(title_umap, fontsize=9)
            ax2.set_xlabel("UMAP1", fontsize=8)
            ax2.set_ylabel("UMAP2", fontsize=8)
            ax2.tick_params(labelsize=7)

    # Hide unused axes
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            idx_check = (row % n_rows_per_method) * n_cols + col
            if idx_check >= n_models:
                axes[row, col].set_visible(False)

    fig.suptitle(
        f"{study_name}: Latent Space Ordination (coloured by {label_col})",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(outdir, "fig1_latent_ordination.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {png_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Classification performance comparison
# ---------------------------------------------------------------------------

def _baseline_classification_records(
    input_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    nmf_components: int = 16,
    results_dir: Optional[str] = None,
    label_col: str = "disease",
) -> List[dict]:
    """Build the NMF + pre-computed XGBoost baseline classification rows.

    The returned records share the per-model schema (``model``,
    ``model_key``, the three metrics and their ``*_std`` columns, plus
    ``n_seeds``) so they can be appended both to the classification figure
    and to ``results_summary.tsv``.  Computing them here once means the
    expensive NMF fit + 5-seed classification is not repeated between the
    figure and the summary table.

    * The NMF baseline (``input_path`` + ``metadata_path`` required) fits
      ``nmf_components`` NMF factors on the raw counts and evaluates the
      same XGBoost classifier on them.
    * The XGBoost baseline (``results_dir`` required) is loaded from the
      pre-computed
      ``<results_dir>/xgboost-baseline/classify/xgboost_baseline_classification_results.json``.
    """
    records: List[dict] = []

    # NMF baseline classification
    if input_path is not None and metadata_path is not None:
        try:
            from biomevae.classify import (
                DEFAULT_EVAL_SEEDS,
                align_embeddings_metadata,
                evaluate_classifiers,
                load_metadata,
            )
            from biomevae.data import load_matrix
            from biomevae.reconstruction import fit_nmf_embeddings

            X_raw, sample_names = load_matrix(input_path, log1p=False)
            # NMF embedding is fit once (with a fixed seed) and then the
            # classifier is evaluated across DEFAULT_EVAL_SEEDS so that the
            # inline NMF baseline shares the 5-seed reproducibility protocol
            # with the rest of the pipeline.
            nmf_emb = fit_nmf_embeddings(
                X_raw, n_components=nmf_components, log1p=True, random_state=42,
            )
            labels, meta_samples = load_metadata(metadata_path, label_col)
            X_clf, y_clf, class_names, _le = align_embeddings_metadata(
                nmf_emb, sample_names, labels, meta_samples,
            )
            seeds = list(DEFAULT_EVAL_SEEDS)
            nmf_results = evaluate_classifiers(
                X_clf, y_clf, class_names,
                n_splits=5, n_repeats=10,
                seeds=seeds,
            )
            # Use only XGBoost results; fall back to first available
            if "XGBoost" in nmf_results:
                res = nmf_results["XGBoost"]
            else:
                res = next(iter(nmf_results.values()))
            nmf_stds = res.across_seed_std or {}
            records.append({
                "model": f"NMF (k={nmf_components})",
                "model_key": "nmf",
                "balanced_accuracy": res.balanced_accuracy,
                "f1_macro": res.f1_macro,
                "auroc": res.auroc,
                "balanced_accuracy_std": nmf_stds.get("balanced_accuracy"),
                "f1_macro_std": nmf_stds.get("f1_macro"),
                "auroc_std": nmf_stds.get("auroc"),
                "n_seeds": len(seeds),
            })
            print(f"  NMF classification: evaluated with XGBoost.")
        except Exception as exc:
            print(f"  WARNING: NMF classification failed: {exc}")

    # XGBoost baseline (direct from SGB table, pre-computed)
    if results_dir is not None:
        baseline_path = (
            Path(results_dir)
            / "xgboost-baseline"
            / "classify"
            / "xgboost_baseline_classification_results.json"
        )
        if baseline_path.exists():
            try:
                baseline_data = _load_json(str(baseline_path))
                if "XGBoost" in baseline_data:
                    bm = baseline_data["XGBoost"]
                else:
                    bm = next(iter(baseline_data.values()))
                bm_stds = bm.get("across_seed_std") or {}
                records.append({
                    "model": "XGBoost (SGB)",
                    "model_key": "xgboost-baseline",
                    "balanced_accuracy": bm["balanced_accuracy"],
                    "f1_macro": bm["f1_macro"],
                    "auroc": bm.get("auroc"),
                    "balanced_accuracy_std": bm_stds.get("balanced_accuracy"),
                    "f1_macro_std": bm_stds.get("f1_macro"),
                    "auroc_std": bm_stds.get("auroc"),
                    "n_seeds": len(bm.get("seeds") or []),
                })
                print("  XGBoost baseline: loaded from pre-computed results.")
            except Exception as exc:
                print(f"  WARNING: XGBoost baseline loading failed: {exc}")
        else:
            print(f"  NOTE: XGBoost baseline not found at {baseline_path}")

    return records


def fig_classification_performance(
    models: Dict[str, Path],
    outdir: str,
    input_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    nmf_components: int = 16,
    results_dir: Optional[str] = None,
    label_col: str = "disease",
    study_name: str = "Study",
    baseline_records: Optional[List[dict]] = None,
) -> Optional[str]:
    """Grouped bar chart of classification metrics across models.

    The NMF and pre-computed XGBoost baselines are appended alongside the
    VAE models.  Pass *baseline_records* to reuse rows already computed by
    :func:`_baseline_classification_records` (avoids re-fitting NMF); when
    omitted they are computed inline from *input_path* / *metadata_path* /
    *results_dir*.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    for name, mdir in models.items():
        clf_path = mdir / "classify" / "classification_results.json"
        if not clf_path.exists():
            continue
        clf_data = _load_json(str(clf_path))
        # Use only XGBoost results; fall back to first available classifier
        if "XGBoost" in clf_data:
            metrics = clf_data["XGBoost"]
        else:
            metrics = next(iter(clf_data.values()))
        stds = metrics.get("across_seed_std") or {}
        records.append({
            "model": _display_name(name),
            "model_key": name,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "auroc": metrics.get("auroc"),
            "balanced_accuracy_std": stds.get("balanced_accuracy"),
            "f1_macro_std": stds.get("f1_macro"),
            "auroc_std": stds.get("auroc"),
        })

    # NMF + pre-computed XGBoost baselines (shared with the summary table).
    if baseline_records is None:
        baseline_records = _baseline_classification_records(
            input_path, metadata_path, nmf_components, results_dir, label_col,
        )
    records.extend(baseline_records)

    if not records:
        print("  No classification results found; skipping.")
        return None

    df = pd.DataFrame(records)

    # Load latent dims and build display labels with dimension info
    latent_dims: Dict[str, Optional[int]] = {}
    for name, mdir in models.items():
        latent_dims[name] = _load_latent_dim(mdir)

    # One record per model (already filtered to XGBoost above)
    df = df.sort_values("balanced_accuracy", ascending=True)

    # Build model labels with latent dim info
    def _label_with_dim(row):
        dim = latent_dims.get(row["model_key"])
        if dim is not None:
            return f"{row['model']}  (d={dim})"
        return row["model"]

    df["label"] = df.apply(_label_with_dim, axis=1)

    metric_cols = ["balanced_accuracy", "f1_macro"]
    if df["auroc"].notna().any():
        metric_cols.append("auroc")

    metric_labels = {
        "balanced_accuracy": "Balanced Accuracy",
        "f1_macro": "F1 (macro)",
        "auroc": "AUROC",
    }

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(4.5 * n_metrics, 0.5 * len(df) + 1.5),
        constrained_layout=True, sharey=True,
    )
    if n_metrics == 1:
        axes = [axes]

    colours = ["#4c72b0", "#dd8452", "#55a868"]
    is_baseline = np.isin(df["model_key"].values, ["nmf", "xgboost-baseline"])

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        vals = df[metric].values.astype(float)
        std_col = f"{metric}_std"
        if std_col in df.columns:
            stds_raw = df[std_col].values
            stds = np.array(
                [float(s) if s is not None and np.isfinite(float(s)) else 0.0
                 for s in stds_raw],
                dtype=float,
            )
        else:
            stds = np.zeros_like(vals)
        bar_colours = [
            "#999999" if bl else colours[i] for bl in is_baseline
        ]
        bars = ax.barh(
            range(len(df)), vals,
            color=bar_colours, edgecolor="white", height=0.6,
            xerr=stds if np.any(stds > 0) else None,
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3},
        )
        ax.set_xlabel(metric_labels[metric], fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["label"].values, fontsize=9)
        ax.tick_params(labelsize=8)

        # Annotate bars
        for bar, val, std in zip(bars, vals, stds):
            if not np.isnan(val):
                # Shift annotation past the error bar cap, when present
                offset = max(0.01, float(std) + 0.01) if std > 0 else 0.01
                if std > 0:
                    label_txt = f"{val:.3f} ± {std:.3f}"
                else:
                    label_txt = f"{val:.3f}"
                ax.text(
                    val + offset, bar.get_y() + bar.get_height() / 2,
                    label_txt, va="center", fontsize=7,
                )

        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        f"{study_name}: {label_col.replace('_', ' ').title()} Classification Performance",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(outdir, "fig2_classification_performance.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {png_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Confusion matrices
# ---------------------------------------------------------------------------

def fig_confusion_matrices(
    models: Dict[str, Path],
    outdir: str,
    study_name: str = "Study",
    label_col: str = "disease",
    results_dir: Optional[str] = None,
) -> Optional[str]:
    """Confusion matrix heatmaps for each model's best classifier.

    When ``results_dir`` is provided, the pre-computed XGBoost baseline
    (direct classification from the SGB table without dimensionality
    reduction) is loaded from
    ``<results_dir>/xgboost-baseline/classify/xgboost_baseline_classification_results.json``
    and appended as an additional panel for direct comparison with the
    VAE-embedding results.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for name, mdir in models.items():
        clf_path = mdir / "classify" / "classification_results.json"
        if not clf_path.exists():
            continue
        clf_data = _load_json(str(clf_path))
        # Use only XGBoost results; fall back to first available classifier
        if "XGBoost" in clf_data:
            best = clf_data["XGBoost"]
        else:
            best = next(iter(clf_data.values()))
        dim = _load_latent_dim(mdir)
        panels.append((name, np.array(best["confusion_matrix"]), best["class_names"], dim))

    # XGBoost baseline (direct from SGB table, pre-computed)
    if results_dir is not None:
        baseline_path = (
            Path(results_dir)
            / "xgboost-baseline"
            / "classify"
            / "xgboost_baseline_classification_results.json"
        )
        if baseline_path.exists():
            try:
                baseline_data = _load_json(str(baseline_path))
                if "XGBoost" in baseline_data:
                    bm = baseline_data["XGBoost"]
                else:
                    bm = next(iter(baseline_data.values()))
                panels.append((
                    "xgboost-baseline",
                    np.array(bm["confusion_matrix"]),
                    bm["class_names"],
                    None,
                ))
                print("  XGBoost baseline confusion matrix: loaded.")
            except Exception as exc:
                print(f"  WARNING: XGBoost baseline confusion matrix loading failed: {exc}")
        else:
            print(f"  NOTE: XGBoost baseline not found at {baseline_path}")

    if not panels:
        print("  No confusion matrices found; skipping.")
        return None

    n = len(panels)
    n_cols = min(n, 4)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (model_name, cm, class_names, latent_dim) in enumerate(panels):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Normalise to percentages
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        dim_str = f"  (d={latent_dim})" if latent_dim is not None else ""
        ax.set_title(f"{_display_name(model_name)}{dim_str}", fontsize=8)

        # Annotate cells
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                txt_colour = "white" if cm_pct[i, j] > 60 else "black"
                ax.text(
                    j, i, f"{cm_pct[i, j]:.0f}%\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color=txt_colour,
                )

    # Hide unused axes
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            if row * n_cols + col >= n:
                axes[row, col].set_visible(False)

    fig.suptitle(
        f"{study_name}: Confusion Matrices ({label_col})",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(outdir, "fig3_confusion_matrices.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {png_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: Reconstruction quality
# ---------------------------------------------------------------------------

def _compute_recon_metrics(
    original: np.ndarray,
    recon: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE and MAE between two arrays of the same shape."""
    diff = original - recon
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
    }


def _load_recon_as_log1p(
    model_dir: Path,
    original_log1p: np.ndarray,
    original_features: List[str],
    original_samples: List[str],
) -> Optional[np.ndarray]:
    """Load a model's reconstruction and convert to log1p count space.

    Returns an array aligned to ``original_features`` ordering, or *None*
    if the reconstruction file is missing.
    """
    # Try embed/recon.tsv first, then test/recon.tsv, then root recon.tsv
    recon_path = model_dir / "embed" / "recon.tsv"
    if not recon_path.exists():
        recon_path = model_dir / "test" / "recon.tsv"
    if not recon_path.exists():
        recon_path = model_dir / "recon.tsv"
    if not recon_path.exists():
        return None

    recon_df = pd.read_csv(str(recon_path), sep="\t", index_col=0)

    # Load config for preprocessing info.
    config_path = model_dir / "config.json"
    cfg: dict = {}
    if config_path.exists():
        with open(config_path) as fh:
            cfg = json.load(fh)

    model_type = cfg.get("model_type", "euclid")
    log1p_flag = bool(cfg.get("log1p", False))
    standardize = bool(cfg.get("standardize", False))

    # Convert recon to log1p-count space.
    recon_arr = recon_df.values.astype(np.float64)

    if model_type in (
        "philrvae", "tree-dtm-vae",
        "hyperbolic-philrvae",
    ):
        # Count-space decoders (NB, Gaussian-on-ILR, Dirichlet, ZINB)
        # output raw-count-scale predictions — apply log1p for fair
        # comparison.
        recon_arr = np.log1p(np.clip(recon_arr, 0, None))
    elif standardize:
        # De-standardise first.
        scaler_path = model_dir / "feature_scaler.npz"
        if scaler_path.exists():
            npz = np.load(str(scaler_path))
            recon_arr = recon_arr * npz["std"] + npz["mean"]
        if not log1p_flag:
            recon_arr = np.log1p(np.clip(recon_arr, 0, None))
    elif not log1p_flag:
        # Raw-space model output — apply log1p.
        recon_arr = np.log1p(np.clip(recon_arr, 0, None))
    # else: already in log1p space.

    # Align features and samples to original ordering.
    recon_cols = list(recon_df.columns)
    recon_samples = list(recon_df.index)

    if model_type == "tree-dtm-vae":
        # Tree-softmax variant (TreeDTM-VAE) may have different
        # features (tree leaves).  Map back
        # to the original feature set where possible, filling with
        # log1p(0)=0 for missing features.
        col_map = {c: i for i, c in enumerate(recon_cols)}
        aligned = np.zeros_like(original_log1p)
        for fi, fname in enumerate(original_features):
            ci = col_map.get(fname)
            if ci is not None:
                for si, sname in enumerate(original_samples):
                    ri = recon_samples.index(sname) if sname in recon_samples else None
                    if ri is not None:
                        aligned[si, fi] = recon_arr[ri, ci]
        return aligned

    # Standard alignment: match samples, assume feature order matches or
    # align by name.
    if set(recon_cols) == set(original_features) and recon_cols != original_features:
        # Same features, different order — reorder.
        recon_df = recon_df[original_features]
        recon_arr = recon_df.values.astype(np.float64)
        if not log1p_flag and model_type not in (
            "philrvae", "tree-dtm-vae",
            "hyperbolic-philrvae",
        ):
            pass  # already converted above
        recon_cols = original_features

    # Align samples.
    sample_map = {s: i for i, s in enumerate(recon_samples)}
    aligned = np.zeros_like(original_log1p)
    n_cols = min(recon_arr.shape[1], aligned.shape[1])
    for si, sname in enumerate(original_samples):
        ri = sample_map.get(sname)
        if ri is not None:
            aligned[si, :n_cols] = recon_arr[ri, :n_cols]
    return aligned


def fig_reconstruction_quality(
    models: Dict[str, Path],
    outdir: str,
    input_path: Optional[str] = None,
    nmf_components: int = 16,
    study_name: str = "Study",
) -> Optional[str]:
    """Bar charts of RMSE and MAE reconstruction metrics with NMF baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if input_path is None:
        print("  WARNING: --input not provided; skipping reconstruction figure.")
        return None

    from biomevae.data import load_matrix
    from biomevae.taxonomy import load_feature_clades

    X_raw, sample_names = load_matrix(input_path, log1p=False)
    X_log1p = np.log1p(X_raw).astype(np.float64)
    feature_names = load_feature_clades(input_path)

    records = []
    latent_dims: Dict[str, Optional[int]] = {}
    for name, mdir in models.items():
        recon_log1p = _load_recon_as_log1p(mdir, X_log1p, feature_names, sample_names)
        if recon_log1p is None:
            continue
        m = _compute_recon_metrics(X_log1p, recon_log1p)
        records.append({
            "model": _display_name(name),
            "model_key": name,
            "rmse": m["rmse"],
            "mae": m["mae"],
        })
        latent_dims[name] = _load_latent_dim(mdir)

    # NMF baseline.
    try:
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=nmf_components, init="nndsvda", random_state=42, max_iter=500)
        W = nmf.fit_transform(X_log1p)
        nmf_recon = W @ nmf.components_
        m = _compute_recon_metrics(X_log1p, nmf_recon)
        records.append({
            "model": f"NMF (k={nmf_components})",
            "model_key": "nmf",
            "rmse": m["rmse"],
            "mae": m["mae"],
        })
    except Exception as exc:
        print(f"  WARNING: NMF baseline failed: {exc}")

    if not records:
        print("  No reconstruction data found; skipping figure.")
        return None

    df = pd.DataFrame(records).sort_values("rmse", ascending=True)

    # Build model labels with latent dim info
    def _label_with_dim(row):
        dim = latent_dims.get(row["model_key"])
        if dim is not None:
            return f"{row['model']}  (d={dim})"
        return row["model"]

    df["label"] = df.apply(_label_with_dim, axis=1)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 0.45 * len(df) + 1.5),
        constrained_layout=True, sharey=True,
    )

    metric_specs = [
        ("rmse", "RMSE (log1p counts)", "#4c72b0"),
        ("mae", "MAE (log1p counts)", "#dd8452"),
    ]

    for i, (col, label, colour) in enumerate(metric_specs):
        ax = axes[i]
        vals = df[col].values.astype(float)
        is_nmf = df["model_key"].values == "nmf"

        bar_colours = [
            "#999999" if nmf else colour for nmf in is_nmf
        ]
        bars = ax.barh(
            range(len(df)), vals,
            color=bar_colours, edgecolor="white", height=0.6,
        )
        ax.set_xlabel(label, fontsize=10)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["label"].values, fontsize=9)
        ax.tick_params(labelsize=8)
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(v + 0.001, j, f"{v:.4f}", va="center", fontsize=7)

    fig.suptitle(
        f"{study_name}: Reconstruction Quality Across Models",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(outdir, "fig4_reconstruction_quality.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {png_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 5: Training curves
# ---------------------------------------------------------------------------

def fig_training_curves(
    models: Dict[str, Path],
    outdir: str,
    study_name: str = "Study",
) -> Optional[str]:
    """Training and validation loss curves for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = {}
    latent_dims: Dict[str, Optional[int]] = {}
    for name, mdir in models.items():
        log_path = mdir / "training_log.tsv"
        if log_path.exists():
            logs[name] = _load_training_log(str(log_path))
            latent_dims[name] = _load_latent_dim(mdir)

    if not logs:
        print("  No training logs found; skipping training curves.")
        return None

    n = len(logs)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (name, log_df) in enumerate(sorted(logs.items())):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        epoch_col = "epoch" if "epoch" in log_df.columns else log_df.columns[0]
        epochs = log_df[epoch_col].values

        # Reconstruction loss is the only *stationary* convergence signal for
        # β-annealed VAEs: it is independent of the KL warmup schedule and
        # therefore comparable across epochs. Show it as the primary, bold
        # solid line. The β-weighted ELBO (``train_loss``/``val_loss``) is
        # informative for completeness but non-stationary during warmup, so
        # plot it as a faint secondary reference.
        plotted_values: list[np.ndarray] = []
        for recon_col, colour, label in [
            ("train_recon", "#4c72b0", "Train recon"),
            ("val_recon", "#d62728", "Val recon"),
        ]:
            if recon_col in log_df.columns:
                vals = log_df[recon_col].values
                ax.plot(
                    epochs, vals, color=colour,
                    label=label, linewidth=1.6,
                )
                plotted_values.append(np.asarray(vals, dtype=float))

        for loss_col, colour, label in [
            ("train_loss", "#4c72b0", "Train ELBO"),
            ("val_loss", "#d62728", "Val ELBO"),
        ]:
            if loss_col in log_df.columns:
                vals = log_df[loss_col].values
                ax.plot(
                    epochs, vals, color=colour,
                    label=label, linewidth=0.7, linestyle=":", alpha=0.45,
                )
                plotted_values.append(np.asarray(vals, dtype=float))

        dim = latent_dims.get(name)
        dim_str = f"  (d={dim})" if dim is not None else ""
        ax.set_title(f"{_display_name(name)}{dim_str}", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Reconstruction loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

        # Log-scale y is preferred (VAE losses span orders of magnitude),
        # but falls back to linear for likelihoods that can be non-positive
        # (e.g. Dirichlet NLL, which is −log density on a high-dim simplex
        # and routinely goes very negative).
        all_vals = np.concatenate(plotted_values) if plotted_values else np.array([])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size > 0 and np.all(finite > 0):
            ax.set_yscale("log")

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            if row * n_cols + col >= n:
                axes[row, col].set_visible(False)

    fig.suptitle(
        f"{study_name}: Training Curves",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(outdir, "fig5_training_curves.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {png_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 6: Summary results table (LaTeX-ready)
# ---------------------------------------------------------------------------

def generate_results_table(
    models: Dict[str, Path],
    outdir: str,
    input_path: Optional[str] = None,
    study_name: str = "Study",
    baseline_records: Optional[List[dict]] = None,
) -> Optional[str]:
    """Generate a summary table of all results as TSV and LaTeX.

    *baseline_records* (the NMF + XGBoost rows from
    :func:`_baseline_classification_records`) are appended so the
    reference baselines appear in ``results_summary.tsv`` alongside the
    VAE models, matching what the classification figure plots.
    """
    # Pre-compute RMSE/MAE if input is available.
    recon_metrics: Dict[str, Dict[str, float]] = {}
    if input_path is not None:
        from biomevae.data import load_matrix
        from biomevae.taxonomy import load_feature_clades

        X_raw, sample_names = load_matrix(input_path, log1p=False)
        X_log1p = np.log1p(X_raw).astype(np.float64)
        feature_names = load_feature_clades(input_path)
        for name, mdir in models.items():
            recon_log1p = _load_recon_as_log1p(mdir, X_log1p, feature_names, sample_names)
            if recon_log1p is not None:
                recon_metrics[name] = _compute_recon_metrics(X_log1p, recon_log1p)

    records = []
    for name, mdir in models.items():
        row: dict = {"model": _display_name(name), "model_key": name}

        # RMSE / MAE
        if name in recon_metrics:
            row["rmse"] = recon_metrics[name]["rmse"]
            row["mae"] = recon_metrics[name]["mae"]

        # Test report
        report_path = mdir / "test" / "test_report.json"
        if report_path.exists():
            report = _load_json(str(report_path))
            row["recon_loss"] = report.get("reconstruction")
            row["kl_divergence"] = report.get("kl_mean")

        # Classification results (XGBoost only)
        clf_path = mdir / "classify" / "classification_results.json"
        if clf_path.exists():
            clf_data = _load_json(str(clf_path))
            if "XGBoost" in clf_data:
                best = clf_data["XGBoost"]
            else:
                best = next(iter(clf_data.values()))
            row["balanced_accuracy"] = best["balanced_accuracy"]
            row["f1_macro"] = best["f1_macro"]
            row["auroc"] = best.get("auroc")
            stds = best.get("across_seed_std") or {}
            row["balanced_accuracy_std"] = stds.get("balanced_accuracy")
            row["f1_macro_std"] = stds.get("f1_macro")
            row["auroc_std"] = stds.get("auroc")
            row["n_seeds"] = len(best.get("seeds") or [])

        # Config
        config_path = mdir / "config.json"
        if config_path.exists():
            cfg = _load_json(str(config_path))
            row["latent_dim"] = cfg.get("latent_dim")
            row["epochs"] = cfg.get("epochs")

        records.append(row)

    # Append the NMF + XGBoost reference baselines so they appear in the
    # summary table alongside the VAE models (these have no model directory
    # of their own, so the loop above never sees them).
    if baseline_records:
        records.extend(baseline_records)

    if not records:
        print("  No results to tabulate; skipping.")
        return None

    df = pd.DataFrame(records)
    df = df.sort_values("balanced_accuracy", ascending=False, na_position="last")

    # Save TSV
    tsv_path = os.path.join(outdir, "results_summary.tsv")
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.4f")
    print(f"  Saved: {tsv_path}")

    # Generate LaTeX table
    latex_path = os.path.join(outdir, "results_summary.tex")
    cols = ["model", "latent_dim", "rmse", "mae",
            "balanced_accuracy", "f1_macro", "auroc"]
    existing_cols = [c for c in cols if c in df.columns]
    latex_df = df[existing_cols].copy()

    header_map = {
        "model": "Model",
        "latent_dim": "$d$",
        "rmse": "RMSE",
        "mae": "MAE",
        "balanced_accuracy": "Bal. Acc.",
        "f1_macro": "F1 (macro)",
        "auroc": "AUROC",
    }
    latex_df = latex_df.rename(columns=header_map)

    with open(latex_path, "w") as f:
        f.write(f"% Auto-generated results table for {study_name} analysis\n")
        f.write(latex_df.to_latex(index=False, float_format="%.4f", na_rep="--"))
    print(f"  Saved: {latex_path}")

    return tsv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-single-study-figures",
        description=(
            "Generate publication-quality figures and results tables for a "
            "single-study biomevae run. Expects a results directory "
            "structured as: <results>/<model-name>/{config.json, model.pt, "
            "embed/embeddings.tsv, test/test_report.json, "
            "classify/classification_results.json, training_log.tsv}. "
            "Data files (sgb_table.tsv, phyla.tsv, sample_metadata.tsv) "
            "should be produced beforehand by the extract-microbiome-data "
            "package."
        ),
    )
    ap.add_argument(
        "--results-dir", required=True,
        help="Root directory containing per-model result subdirectories.",
    )
    ap.add_argument(
        "--metadata", required=True,
        help="Path to sample_metadata.tsv.",
    )
    ap.add_argument(
        "--outdir", required=True,
        help="Output directory for figures and tables.",
    )
    ap.add_argument(
        "--input", default=None,
        help=(
            "Path to the original counts matrix (sgb_table.tsv). "
            "Required for RMSE/MAE reconstruction metrics and NMF baseline."
        ),
    )
    ap.add_argument(
        "--nmf-components", type=int, default=16,
        help="Number of NMF components for baseline (default: 16).",
    )
    ap.add_argument(
        "--label", default="disease",
        help=(
            "Metadata column used for colouring ordination points "
            "(default: disease). Should match the column used for classification."
        ),
    )
    ap.add_argument(
        "--skip-ordination", action="store_true",
        help="Skip latent space ordination figure (can be slow).",
    )
    ap.add_argument(
        "--study-name", default="Study",
        help=(
            "Short identifier used in figure titles and the LaTeX results "
            "table header (default: 'Study')."
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print(f"biomevae-single-study-figures: Publication Figure Generation ({args.study_name})")
    print("=" * 70)

    models = _discover_models(args.results_dir)
    print(f"Discovered {len(models)} model(s): {', '.join(models.keys())}")

    if not models:
        print("ERROR: No trained models found. Check --results-dir.")
        return

    # Compute the NMF + pre-computed XGBoost reference baselines once and
    # share them between the classification figure and the summary table,
    # so the NMF fit is not repeated and both outputs stay in sync.
    print("\n[Baselines] NMF + XGBoost reference classification...")
    baseline_records = _baseline_classification_records(
        input_path=args.input,
        metadata_path=args.metadata,
        nmf_components=args.nmf_components,
        results_dir=args.results_dir,
        label_col=args.label,
    )

    # Figure 1: Latent ordination
    if not args.skip_ordination:
        print("\n[Figure 1] Latent space ordination...")
        fig_latent_ordination(
            models, args.metadata, args.outdir,
            label_col=args.label,
            study_name=args.study_name,
        )
    else:
        print("\n[Figure 1] Skipped (--skip-ordination).")

    # Figure 2: Classification performance (VAE embeddings + NMF baseline)
    print("\n[Figure 2] Classification performance...")
    fig_classification_performance(
        models, args.outdir,
        input_path=args.input,
        metadata_path=args.metadata,
        nmf_components=args.nmf_components,
        results_dir=args.results_dir,
        label_col=args.label,
        study_name=args.study_name,
        baseline_records=baseline_records,
    )

    # Figure 3: Confusion matrices
    print("\n[Figure 3] Confusion matrices...")
    fig_confusion_matrices(
        models, args.outdir,
        study_name=args.study_name,
        label_col=args.label,
        results_dir=args.results_dir,
    )

    # Figure 4: Reconstruction quality (RMSE / MAE with NMF baseline)
    print("\n[Figure 4] Reconstruction quality...")
    fig_reconstruction_quality(
        models, args.outdir, args.input, args.nmf_components,
        study_name=args.study_name,
    )

    # Figure 5: Training curves
    print("\n[Figure 5] Training curves...")
    fig_training_curves(models, args.outdir, study_name=args.study_name)

    # Results table
    print("\n[Table] Summary results...")
    generate_results_table(
        models, args.outdir, args.input, study_name=args.study_name,
        baseline_records=baseline_records,
    )

    print("\n" + "=" * 70)
    print("Figure generation complete.")
    print(f"All outputs saved to: {args.outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
