"""Interpret VAE embeddings in terms of original OTU variables."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_regression

from biomevae.taxonomy import (
    TAX_LEVELS_ALL,
    build_phylo_embeddings,
    build_taxonomy_graph_from_taxonomy,
    load_feature_clades,
    load_taxonomy_table,
)

shap: Any = None  # lazy import; loaded in main()


@dataclass
class ModelArtifacts:
    config: dict
    scaler: dict | None


def _load_counts_table(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load the microbiome table keeping OTU metadata."""

    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.shape[1] < 3:
        raise SystemExit(
            "Expected >=3 columns (clade_name, NCBI_tax_id, samples...) in the input table"
        )

    sample_columns = df.columns[2:]
    otu_names = df.iloc[:, 0].astype(str).tolist()

    values = (
        df[sample_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        .T
    )  # [samples, features]

    sample_names = list(sample_columns)
    return values, sample_names, otu_names


def _load_artifacts(model_dir: str) -> ModelArtifacts:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as handle:
        config = json.load(handle)

    scaler_path = os.path.join(model_dir, "feature_scaler.npz")
    scaler: dict | None
    if config.get("standardize", False) and os.path.exists(scaler_path):
        arr = np.load(scaler_path)
        scaler = {"mean": arr["mean"], "std": arr["std"]}
    else:
        scaler = None

    return ModelArtifacts(config=config, scaler=scaler)


def _apply_scaler(X: np.ndarray, scaler: dict | None) -> np.ndarray:
    if scaler is None:
        return X
    mean = scaler["mean"]
    std = scaler["std"].copy()
    std[std == 0.0] = 1.0
    return ((X - mean) / std).astype(np.float32)


def _build_model(
    cfg: dict,
    input_dim: int,
    *,
    taxonomy_path: str | None = None,
    feature_clades: Sequence[str] | None = None,
) -> torch.nn.Module:
    model_type = cfg.get("model_type", "euclid")
    kwargs = dict(cfg.get("model_kwargs", {}))
    kwargs.pop("graph_spec", None)
    kwargs.pop("phylo_embeddings", None)
    if model_type == "hyperbolic":
        from biomevae.models.hyperbolic import HyperbolicVAE as Model
    elif model_type == "graph_tax":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-interpret: --taxonomy is required for graph_tax models.")
        from biomevae.models.graph import TaxonomyGraphVAE as Model, prepare_graph_kwargs

        mode = kwargs.get("graph_mode", "unweighted")
        graph_spec = build_taxonomy_graph_from_taxonomy(feature_clades, taxonomy_path, mode=mode)
        kwargs = prepare_graph_kwargs({**kwargs, "graph_spec": graph_spec})
    elif model_type == "treeprior":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-interpret: --taxonomy is required for treeprior models.")
        from biomevae.models.treeprior import TreeStructuredPriorVAE as Model, prepare_tree_kwargs

        mode = kwargs.get("graph_mode", "unweighted")
        graph_spec = build_taxonomy_graph_from_taxonomy(feature_clades, taxonomy_path, mode=mode)
        kwargs = prepare_tree_kwargs({**kwargs, "graph_spec": graph_spec})
    elif model_type == "phylo_fusion":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-interpret: --taxonomy is required for phylo_fusion models.")
        from biomevae.models.phylo_fusion import DeepPhyloFusionVAE as Model, prepare_fusion_kwargs

        method = kwargs.get("phylo_method", "pca")
        dim = int(kwargs.get("phylo_dim", 32))
        phylo = build_phylo_embeddings(feature_clades, taxonomy_path, method=method, dim=dim)
        kwargs = prepare_fusion_kwargs({**kwargs, "phylo_embeddings": phylo})
    elif model_type == "tree-dtm-vae":
        if taxonomy_path is None:
            raise SystemExit(
                "biomevae-interpret: --taxonomy is required for tree-dtm-vae models."
            )
        from pathlib import Path

        from biomevae.models.taxonomy_tree import build_taxonomy_graph_from_phyla_tsv
        from biomevae.models.tree_dtm_vae import TreeDTMVAE, build_tree_topology

        taxg = build_taxonomy_graph_from_phyla_tsv(
            Path(taxonomy_path),
            keep_prefixes=bool(kwargs.get("keep_prefixes", False)),
            has_header=bool(kwargs.get("taxonomy_has_header", False)),
            on_duplicate_leaf="ignore_same",
        )
        topo = build_tree_topology(taxg)
        return TreeDTMVAE(
            topo,
            hidden=int(cfg.get("hidden", 256)),
            latent_dim=int(cfg["latent_dim"]),
            encoder_layers=int(cfg.get("encoder_layers", 2)),
            decoder_hidden=int(cfg.get("decoder_hidden", 256)),
            decoder_layers=int(cfg.get("decoder_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            encoder_pseudocount=float(cfg.get("encoder_pseudocount", 0.5)),
            init_concentration=float(cfg.get("init_concentration", 50.0)),
            likelihood=cfg.get("likelihood", "dirichlet_tree_multinomial"),
        )
    elif model_type in ("philrvae", "hyperbolic-philrvae"):
        if taxonomy_path is None:
            raise SystemExit(
                f"biomevae-interpret: --taxonomy is required for {model_type} models."
            )
        from biomevae.models.taxonomy_tree import build_taxonomy_graph_from_phyla_tsv
        taxg = build_taxonomy_graph_from_phyla_tsv(
            Path(taxonomy_path),
            keep_prefixes=bool(kwargs.get("keep_prefixes", False)),
            has_header=bool(kwargs.get("taxonomy_has_header", False)),
            on_duplicate_leaf="ignore_same",
        )
        common = dict(
            latent_dim=int(cfg["latent_dim"]),
            hidden=tuple(cfg.get("hidden", [256, 128])),
            dropout=float(cfg.get("dropout", 0.1)),
            count_pseudocount=float(cfg.get("count_pseudocount", 0.5)),
            relative_pseudocount=float(cfg.get("relative_pseudocount", 1e-6)),
            default_likelihood=cfg.get("likelihood", "philr_gaussian"),
            init_coord_scale=float(cfg.get("init_coord_scale", 0.5)),
            init_concentration=float(cfg.get("init_concentration", 50.0)),
        )
        if model_type == "hyperbolic-philrvae":
            from biomevae.models.hyperbolic_philrvae import HyperbolicPhILRVAE
            return HyperbolicPhILRVAE(
                taxg, curvature=float(cfg.get("curvature", 1.0)), **common,
            )
        from biomevae.models.philrvae import PhILRVAE
        return PhILRVAE(taxg, **common)
    elif model_type == "dsvae":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit(
                "biomevae-interpret: --taxonomy is required for dsvae models."
            )
        from biomevae.models.dsvae import DSVAE
        from biomevae.models.tree_spec import TreeSpec, build_tree_spec as _build_ts

        ts_json = cfg.get("tree_spec")
        if ts_json:
            tree_spec = TreeSpec.from_json(ts_json)
        else:
            branchlen = cfg.get("branchlen_mode", "unit")
            tree_spec = _build_ts(feature_clades, taxonomy_path, branchlen_mode=branchlen)
        supervised = bool(cfg.get("supervised", False))
        n_classes = int(cfg["n_classes"]) if supervised and cfg.get("n_classes") else None
        return DSVAE(
            n_features=input_dim,
            latent_dim=int(cfg["latent_dim"]),
            tree_spec=tree_spec,
            supervised=supervised,
            n_classes=n_classes,
            hidden=list(cfg.get("hidden", [512, 256, 128])),
            dropout=float(cfg.get("dropout", 0.1)),
            pseudocount=float(cfg.get("pseudocount", 0.5)),
            classifier_hidden=int(cfg.get("classifier_hidden", 128)),
        )
    elif model_type == "capda-vae":
        # CAPDA rebuilds its network purely from the saved config dims; the
        # raw species counts SHAP perturbs are expanded to the multi-resolution
        # VAE input by CAPDARawEncoderMean below.
        from biomevae.models.capda_vae import build_capda_from_config
        return build_capda_from_config(cfg)
    elif model_type == "euclid":
        from biomevae.models.vae import VAE as Model
    else:
        raise SystemExit(f"biomevae-interpret: unsupported model_type '{model_type}'.")

    return Model(
        input_dim=input_dim,
        hidden=cfg["hidden"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
        layer_norm=cfg["layer_norm"],
        **kwargs,
    )


class EncoderMean(torch.nn.Module):
    """Wrapper that exposes the encoder mean (µ) as a PyTorch module."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        mu, _ = self.model.encode(inputs)
        return mu


class TreeDTMEncoderMean(torch.nn.Module):
    """Wrapper that reorders flat leaf-count vectors for TreeDTMVAE.

    SHAP's KernelExplainer perturbs flat feature vectors in the input
    column order (``otu_names``).  This wrapper reorders them to the
    taxonomy leaf ordering, aggregates leaf values to all tree nodes,
    and calls ``model.encode()``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        taxg: Any,
        otu_names: Sequence[str],
    ):
        super().__init__()
        self.model = model
        n_leaves = len(taxg.leaf_ids)
        n_nodes = len(taxg.node_names)
        col_index = {name: j for j, name in enumerate(otu_names)}
        perm = []
        for nid in taxg.leaf_ids:
            name = taxg.node_names[nid]
            perm.append(col_index.get(name, -1))
        self.register_buffer("_perm", torch.tensor(perm, dtype=torch.long))
        leaf_ids = torch.tensor(list(taxg.leaf_ids), dtype=torch.long)
        self.register_buffer("_leaf_ids", leaf_ids)

        # Build a leaves->nodes aggregation matrix once. A[node, leaf] == 1
        # iff ``leaf`` descends from ``node``.
        import numpy as _np
        A = _np.zeros((n_nodes, n_leaves), dtype=_np.float32)
        parent = taxg.parent_of
        for li, lid in enumerate(taxg.leaf_ids):
            cur = int(lid)
            while True:
                A[cur, li] = 1.0
                if cur not in parent:
                    break
                cur = int(parent[cur])
        self.register_buffer("_leaf_to_node", torch.from_numpy(A))
        self._n_leaves = n_leaves
        self._n_nodes = n_nodes

    def forward(self, leaf_counts: torch.Tensor) -> torch.Tensor:
        """``leaf_counts``: ``(batch, n_features)`` → ``mu``: ``(batch, latent_dim)``."""
        reordered = torch.zeros(
            leaf_counts.shape[0], self._n_leaves,
            device=leaf_counts.device, dtype=leaf_counts.dtype,
        )
        valid = self._perm >= 0
        reordered[:, valid] = leaf_counts[:, self._perm[valid]]
        node_values = reordered @ self._leaf_to_node.t()
        mu, _ = self.model.encode(node_values)
        return mu


def _select_subset(
    n: int, size: int | None, rng: np.random.Generator, *, allow_all: bool = True
) -> np.ndarray:
    if size is None or (allow_all and size >= n):
        return np.arange(n)
    size = max(1, min(size, n))
    return rng.choice(n, size=size, replace=False)


def _compute_shap_values(
    encoder: EncoderMean,
    background: torch.Tensor,
    explain_samples: torch.Tensor,
    latent_dim: int,
    nsamples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return SHAP values and expected values for each latent dimension."""

    def _to_numpy(obj: object) -> np.ndarray:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return np.asarray(obj, dtype=np.float32)

    def _stack(values: object) -> np.ndarray:
        if isinstance(values, list):
            return np.stack([_to_numpy(v) for v in values], axis=0)
        return _to_numpy(values)

    encoder.eval()

    device = background.device
    background_np = background.detach().cpu().numpy()
    explain_np = explain_samples.detach().cpu().numpy()

    print(
        "[biomevae-interpret] Using KernelExplainer",
        f"background_shape={background_np.shape}",
        f"explain_shape={explain_np.shape}",
        f"latent_dim={latent_dim}",
        f"nsamples={nsamples}",
    )

    def _predict(batch: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(batch.astype(np.float32)).to(device)
        with torch.no_grad():
            outputs = encoder(xt)
        outputs_np = outputs.detach().cpu().numpy()
        print(
            "[biomevae-interpret] _predict",
            f"batch_shape={batch.shape}",
            f"output_shape={outputs_np.shape}",
        )
        return outputs_np

    kernel_explainer = shap.KernelExplainer(_predict, background_np)
    kernel_values = kernel_explainer.shap_values(explain_np, nsamples=nsamples)
    kernel_expected = kernel_explainer.expected_value

    shap_array = _ensure_latent_first(_stack(kernel_values), latent_dim)
    expected_arr = _to_numpy(kernel_expected)

    return shap_array, expected_arr


def _summarize_background(
    background: np.ndarray, method: str, size: int
) -> np.ndarray:
    """Reduce the background set using SHAP helpers."""

    size = max(1, min(size, background.shape[0]))
    if method == "sample":
        summary = shap.sample(background, size)
    else:
        summary = shap.kmeans(background, size)

    data: Any
    if hasattr(summary, "data"):
        data = summary.data
    else:
        data = summary

    summary_array = np.asarray(data, dtype=np.float32)
    if summary_array.ndim != 2:
        raise SystemExit(
            "Summarized background must be 2-D [samples, features]; "
            f"received shape {summary_array.shape}."
        )

    print(
        "[biomevae-interpret] Summarized background",
        f"method={method}",
        f"target_size={size}",
        f"result_shape={summary_array.shape}",
    )
    return summary_array


def _ensure_latent_first(
    shap_values: np.ndarray, latent_dim: int
) -> np.ndarray:
    """Coerce SHAP array to [latent, samples, features] layout."""

    print(
        "[biomevae-interpret] _ensure_latent_first",
        f"initial_shape={getattr(shap_values, 'shape', None)}",
        f"latent_dim={latent_dim}",
    )

    if shap_values.ndim == 2:
        # Single latent dimension may return [samples, features]
        if latent_dim != 1:
            print(
                "[biomevae-interpret] unexpected 2D SHAP values",
                f"shape={shap_values.shape}",
            )
            raise SystemExit(
                "Unexpected SHAP output shape. Expected [latent, samples, features]."
            )
        shap_values = shap_values[None, ...]

    if shap_values.ndim != 3:
        print(
            "[biomevae-interpret] invalid SHAP ndim",
            f"ndim={shap_values.ndim}",
            f"shape={shap_values.shape}",
        )
        raise SystemExit(
            "Unexpected SHAP output shape. Expected [latent, samples, features]."
        )

    if shap_values.shape[0] == latent_dim:
        return shap_values
    if shap_values.shape[1] == latent_dim:
        print(
            "[biomevae-interpret] transposing SHAP values",
            f"current_shape={shap_values.shape}",
            "swap=(1, 0, 2)",
        )
        return np.transpose(shap_values, (1, 0, 2))
    if shap_values.shape[2] == latent_dim:
        print(
            "[biomevae-interpret] transposing SHAP values",
            f"current_shape={shap_values.shape}",
            "swap=(2, 0, 1)",
        )
        return np.transpose(shap_values, (2, 0, 1))

    print(
        "[biomevae-interpret] SHAP latent mismatch",
        f"shape={shap_values.shape}",
        f"latent_dim={latent_dim}",
    )
    raise SystemExit(
        f"SHAP output has {shap_values.shape[0]} latent dimensions, expected {latent_dim}."
    )


def _normalize_expected_values(expected: np.ndarray, latent_dim: int) -> np.ndarray:
    """Ensure expected value vector has shape [latent]."""

    expected = np.asarray(expected, dtype=np.float32)

    if expected.ndim == 0:
        expected = expected.reshape(1)

    if expected.ndim == 1:
        if expected.shape[0] == latent_dim:
            return expected
        if latent_dim == 1 and expected.shape[0] == 1:
            return expected

    axes = [axis for axis, size in enumerate(expected.shape) if size == latent_dim]
    if axes:
        expected = np.moveaxis(expected, axes[0], 0)
        return expected.reshape(latent_dim, -1).mean(axis=1)

    raise SystemExit(
        f"Expected {latent_dim} SHAP expected values, found shape {expected.shape}"
    )


def _compute_spearman(
    embeddings: np.ndarray, features: np.ndarray
) -> np.ndarray:
    """Spearman correlation between latent dimensions and OTU features."""

    emb_rank = pd.DataFrame(embeddings).rank(method="average").to_numpy(dtype=np.float32).copy()
    feat_rank = pd.DataFrame(features).rank(method="average").to_numpy(dtype=np.float32).copy()

    emb_rank -= emb_rank.mean(axis=0, keepdims=True)
    feat_rank -= feat_rank.mean(axis=0, keepdims=True)

    numerator = feat_rank.T @ emb_rank
    emb_var = np.sqrt((emb_rank**2).sum(axis=0))
    feat_var = np.sqrt((feat_rank**2).sum(axis=0))

    denom = np.outer(feat_var, emb_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denom
    corr = np.nan_to_num(corr, nan=0.0)
    return corr  # shape [features, latent]


def _write_dataframe(path: str, values: np.ndarray, index: Sequence[str], columns: Sequence[str]) -> None:
    df = pd.DataFrame(values, index=index, columns=columns)
    df.to_csv(path, sep="\t")


def _plot_topk_bars(
    outdir: str,
    values_matrix: np.ndarray,
    otu_names: Sequence[str],
    latent_names: Sequence[str],
    top_k: int,
    *,
    value_label: str,
    title_prefix: str,
    filename_prefix: str,
    bar_color: str,
) -> None:
    import matplotlib.pyplot as plt

    if values_matrix.ndim != 2:
        raise SystemExit(
            f"Expected 2-D values for {title_prefix} barplots; "
            f"received shape {values_matrix.shape}."
        )
    if values_matrix.shape[1] != len(otu_names):
        raise SystemExit(
            f"{title_prefix} values have {values_matrix.shape[1]} features, "
            f"but {len(otu_names)} feature names were provided."
        )

    for latent_idx, latent_name in enumerate(latent_names):
        row = values_matrix[latent_idx]
        # VAEs commonly collapse a subset of latent dimensions (posterior
        # collapse): for those, SHAP / mutual-info / spearman produce all
        # zeros or non-finite values.  We skip the plot for that dim but
        # keep going so downstream artifacts (summary TSV) still get written.
        if not np.isfinite(row).any():
            print(
                f"[biomevae-interpret] {title_prefix} values for {latent_name} "
                "contain no finite entries; skipping barplot.",
                flush=True,
            )
            continue
        if np.nanmax(np.abs(row)) <= 0.0:
            print(
                f"[biomevae-interpret] {title_prefix} values for {latent_name} "
                "are all zeros (likely a collapsed latent dim); skipping barplot.",
                flush=True,
            )
            continue
        order = np.argsort(row)[::-1][:top_k]
        values = row[order]
        labels = [otu_names[idx] for idx in order]
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * top_k), 4))
        y_pos = np.arange(len(values))
        ax.barh(y_pos, values[::-1], color=bar_color)
        ax.set_yticks(y_pos, labels[::-1])
        ax.set_xlabel(value_label)
        ax.set_title(f"{title_prefix}: Top {top_k} features for {latent_name}")
        fig.tight_layout()
        output_path = os.path.join(outdir, f"{filename_prefix}_{latent_name}.png")
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def _plot_heatmap(
    outdir: str,
    filename: str,
    title: str,
    values: np.ndarray,
    latent_names: Sequence[str],
    feature_names: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(feature_names)), 4 + 0.4 * len(latent_names)))
    mesh = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(latent_names)), labels=latent_names)
    x_positions = np.arange(len(feature_names))
    max_labels = 30
    if len(feature_names) > max_labels:
        stride = int(np.ceil(len(feature_names) / max_labels))
        x_positions = x_positions[::stride]
    ax.set_xticks(x_positions, labels=[feature_names[idx] for idx in x_positions])
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close(fig)


def _select_overall_top_features(
    values: np.ndarray,
    otu_names: Sequence[str],
    top_k: int,
) -> List[str]:
    if values.ndim != 2:
        raise SystemExit(
            f"Expected a 2-D feature matrix for overall selection; got {values.shape}."
        )
    if values.shape[1] != len(otu_names):
        raise SystemExit(
            f"Feature matrix has {values.shape[1]} features, "
            f"but {len(otu_names)} feature names were provided."
        )
    if not np.isfinite(values).any():
        print(
            "[biomevae-interpret] feature matrix contains no finite values; "
            "falling back to the first features for overall selection.",
            flush=True,
        )
        return list(otu_names[:top_k])
    if np.nanmax(np.abs(values)) <= 0.0:
        print(
            "[biomevae-interpret] feature matrix contains only zeros; "
            "falling back to the first features for overall selection.",
            flush=True,
        )
        return list(otu_names[:top_k])
    order = np.argsort(np.nanmean(np.abs(values), axis=0))[::-1][:top_k]
    return [otu_names[idx] for idx in order]


def _summarize_top_features(
    otu_names: Sequence[str],
    latent_names: Sequence[str],
    shap_mean_abs: np.ndarray,
    shap_mean: np.ndarray,
    spearman: np.ndarray,
    mutual_info: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    records = []
    for latent_idx, latent_name in enumerate(latent_names):
        order = np.argsort(shap_mean_abs[latent_idx])[::-1][:top_k]
        for rank, feat_idx in enumerate(order, start=1):
            records.append(
                {
                    "latent": latent_name,
                    "rank": rank,
                    "otu": otu_names[feat_idx],
                    "mean_abs_shap": float(shap_mean_abs[latent_idx, feat_idx]),
                    "mean_shap": float(shap_mean[latent_idx, feat_idx]),
                    "spearman": float(spearman[feat_idx, latent_idx]),
                    "mutual_info": float(mutual_info[latent_idx, feat_idx]),
                }
            )

    return pd.DataFrame(records)


def _normalize_taxonomy_level(level: str) -> str:
    norm = level.strip().lower()
    if norm in TAX_LEVELS_ALL:
        return norm
    aliases = {
        "kingdom": "k",
        "phylum": "p",
        "class": "c",
        "order": "o",
        "family": "f",
        "genus": "g",
        "species": "s",
    }
    if norm in aliases:
        return aliases[norm]
    raise SystemExit(
        f"Unknown taxonomy level '{level}'. Choose among {TAX_LEVELS_ALL} "
        "or full names (kingdom/phylum/class/order/family/genus/species)."
    )


def _build_taxonomy_aggregation(
    otu_names: Sequence[str],
    taxonomy_path: str,
    level: str,
) -> Tuple[np.ndarray, List[str]]:
    tax = load_taxonomy_table(taxonomy_path)
    tax_aligned = tax.reindex(otu_names)
    if tax_aligned.isna().any().any():
        tax_aligned = tax_aligned.fillna({lvl: f"NA_{lvl}" for lvl in TAX_LEVELS_ALL})

    labels = tax_aligned[level].astype(str).to_numpy()
    group_labels = pd.unique(labels).tolist()
    index = {label: idx for idx, label in enumerate(group_labels)}

    A = np.zeros((len(group_labels), len(otu_names)), dtype=np.float32)
    for feat_idx, label in enumerate(labels):
        A[index[label], feat_idx] = 1.0
    return A, group_labels


def _aggregate_feature_matrix(values: np.ndarray, agg_matrix: np.ndarray) -> np.ndarray:
    return values @ agg_matrix.T


def _aggregate_shap_values(
    shap_values: np.ndarray, agg_matrix: np.ndarray
) -> np.ndarray:
    latent, samples, features = shap_values.shape
    flattened = shap_values.reshape(latent * samples, features)
    aggregated = flattened @ agg_matrix.T
    return aggregated.reshape(latent, samples, agg_matrix.shape[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("biomevae-interpret")
    parser.add_argument("--input", required=True, help="Counts table used for training (TSV)")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt and config.json")
    parser.add_argument("--outdir", required=True, help="Output directory for interpretation artifacts")
    parser.add_argument("--background-size", type=int, default=128, help="Samples to use as SHAP background")
    parser.add_argument(
        "--background-summary",
        choices=["sample", "kmeans"],
        help="Summarize the SHAP background with shap.sample or shap.kmeans",
    )
    parser.add_argument(
        "--background-summary-size",
        type=int,
        help="Samples to keep after summarizing the background (default: background-size)",
    )
    parser.add_argument(
        "--explain-size",
        type=int,
        default=256,
        help="Number of samples for SHAP attribution (default: min(256, n_samples))",
    )
    parser.add_argument(
        "--shap-nsamples",
        type=int,
        default=1000,
        help="Number of samples for KernelExplainer; lower values reduce runtime.",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Top features to report per latent dimension")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the encoder and SHAP explainer",
    )
    parser.add_argument("--taxonomy", default=None, help="Optional taxonomy table (TSV/CSV) for aggregation")
    parser.add_argument(
        "--taxonomy-level",
        default=None,
        help="Aggregate features to a taxonomy level (k/p/c/o/f/g/s or full names).",
    )
    parser.add_argument(
        "--save-sample-shap",
        action="store_true",
        help="Save the full SHAP tensor (potentially large) as shap_values.npz",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    global shap
    try:
        import shap as _shap  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "shap is required for embedding interpretation. Install with "
            "`pip install biomevae[interpret]` or add shap to your environment."
        ) from exc
    shap = _shap

    args = build_parser().parse_args(argv)

    if args.shap_nsamples <= 0:
        raise SystemExit("--shap-nsamples must be a positive integer.")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    X_raw, sample_names, otu_names = _load_counts_table(args.input)
    artifacts = _load_artifacts(args.model_dir)
    feature_clades = load_feature_clades(args.input)
    model_type = artifacts.config.get("model_type", "euclid")
    expected_clades = artifacts.config.get("feature_clades")
    # tree-dtm-vae / hgvae_zi save feature_clades in tree-traversal (leaf)
    # order, which differs from the sgb_table row order returned by
    # ``load_feature_clades``. Skip the strict ordering check for these – the
    # TreeDTMEncoderMean wrapper below handles the permutation to leaf order.
    _tree_orderings = {"tree-dtm-vae", "hgvae_zi"}
    if (
        expected_clades
        and model_type not in _tree_orderings
        and list(expected_clades) != list(feature_clades)
    ):
        raise SystemExit("Input feature ordering does not match the trained model.")
    device = torch.device(args.device)

    # -- build model ----------------------------------------------------------
    if model_type == "tree-dtm-vae":
        # Tree-softmax variants operate on raw counts; no log1p / scaler.
        X_model = X_raw.astype(np.float32)
    elif model_type == "capda-vae":
        # SHAP perturbs raw per-species counts; CAPDARawEncoderMean rebuilds the
        # multi-resolution VAE input + scaler inside the encoder wrapper.
        X_model = X_raw.astype(np.float32)
    elif model_type in (
        "philrvae", "hyperbolic-philrvae", "dsvae",
    ):
        # PhILR family and DSVAE operate on raw counts; no log1p / scaler.
        X_model = X_raw.astype(np.float32)
    else:
        X_log = np.log1p(X_raw).astype(np.float32)
        if artifacts.config.get("log1p", False):
            X_model = X_log
        else:
            X_model = X_raw.astype(np.float32)
        X_model = _apply_scaler(X_model, artifacts.scaler)

    model = _build_model(
        artifacts.config,
        input_dim=X_model.shape[1],
        taxonomy_path=args.taxonomy,
        feature_clades=feature_clades,
    ).to(device)
    state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # -- encoder wrapper ------------------------------------------------------
    # Tree-softmax variant (TreeDTMVAE) takes node values, so wrap the
    # raw leaf counts via TreeDTMEncoderMean.
    if model_type == "tree-dtm-vae":
        from pathlib import Path

        from biomevae.models.taxonomy_tree import build_taxonomy_graph_from_phyla_tsv

        taxg = build_taxonomy_graph_from_phyla_tsv(
            Path(args.taxonomy),
            keep_prefixes=bool(
                artifacts.config.get("model_kwargs", {}).get("keep_prefixes", False)
            ),
            has_header=bool(
                artifacts.config.get("model_kwargs", {}).get("taxonomy_has_header", False)
            ),
            on_duplicate_leaf="ignore_same",
        )
        encoder = TreeDTMEncoderMean(model, taxg, otu_names).to(device)
    elif model_type == "capda-vae":
        if args.taxonomy is None:
            raise SystemExit(
                "biomevae-interpret: --taxonomy is required for capda-vae models."
            )
        from biomevae.models.capda_vae import (
            CAPDARawEncoderMean, load_lineage_table,
        )

        taxonomy = load_lineage_table(
            args.taxonomy,
            has_header=bool(artifacts.config.get("taxonomy_has_header", False)),
        )
        encoder = CAPDARawEncoderMean(
            model, feature_clades, taxonomy, artifacts.config,
        ).to(device)
    else:
        encoder = EncoderMean(model).to(device)

    xt = torch.from_numpy(X_model).to(device)
    with torch.no_grad():
        embeddings = encoder(xt).cpu().numpy()

    latent_dim = embeddings.shape[1]
    latent_names = [f"z{i}" for i in range(latent_dim)]

    bg_idx = _select_subset(X_model.shape[0], args.background_size, rng)
    explain_idx = _select_subset(X_model.shape[0], args.explain_size, rng)

    background_np = X_model[bg_idx].astype(np.float32)
    if args.background_summary:
        summary_size = args.background_summary_size or args.background_size
        background_np = _summarize_background(
            background_np,
            args.background_summary,
            summary_size,
        )
    background = torch.from_numpy(background_np).to(device)
    explain = torch.from_numpy(X_model[explain_idx]).to(device)

    shap_values, expected = _compute_shap_values(
        encoder,
        background,
        explain,
        latent_dim,
        args.shap_nsamples,
    )

    agg_matrix = None
    if args.taxonomy_level:
        if not args.taxonomy:
            raise SystemExit("--taxonomy is required when using --taxonomy-level.")
        level = _normalize_taxonomy_level(args.taxonomy_level)
        agg_matrix, otu_names = _build_taxonomy_aggregation(
            otu_names,
            args.taxonomy,
            level,
        )
        X_model = _aggregate_feature_matrix(X_model, agg_matrix)
        shap_values = _aggregate_shap_values(shap_values, agg_matrix)

    shap_mean_abs = np.mean(np.abs(shap_values), axis=1)
    shap_mean = np.mean(shap_values, axis=1)

    mutual_info = []
    for latent_idx in range(latent_dim):
        y = embeddings[:, latent_idx]
        # mutual_info_regression expects non-constant targets; if the latent
        # dim collapsed (variance ~ 0) we just emit zeros rather than
        # propagating NaNs or crashing the whole interpret job.
        if not np.all(np.isfinite(y)) or float(np.nanstd(y)) <= 1e-12:
            scores = np.zeros(X_model.shape[1], dtype=np.float32)
        else:
            try:
                scores = mutual_info_regression(X_model, y, random_state=args.seed)
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    "[biomevae-interpret] mutual_info_regression failed for "
                    f"z{latent_idx}: {exc}. Falling back to zeros.",
                    flush=True,
                )
                scores = np.zeros(X_model.shape[1], dtype=np.float32)
        mutual_info.append(scores)
    mutual_info_matrix = np.stack(mutual_info, axis=0)

    spearman_matrix = _compute_spearman(embeddings, X_model)

    _plot_topk_bars(
        args.outdir,
        shap_mean_abs,
        otu_names,
        latent_names,
        args.top_k,
        value_label="Mean |SHAP|",
        title_prefix="SHAP",
        filename_prefix="shap_top_features",
        bar_color="#4C78A8",
    )
    _plot_topk_bars(
        args.outdir,
        mutual_info_matrix,
        otu_names,
        latent_names,
        args.top_k,
        value_label="Mutual information",
        title_prefix="Feature importance",
        filename_prefix="feature_importance_top_features",
        bar_color="#F58518",
    )

    shap_features = _select_overall_top_features(shap_mean_abs, otu_names, args.top_k)
    shap_indices = [otu_names.index(name) for name in shap_features]
    _plot_heatmap(
        args.outdir,
        "shap_mean_abs_heatmap.png",
        "Mean |SHAP| per latent dimension",
        shap_mean_abs[:, shap_indices],
        latent_names,
        shap_features,
    )

    mi_features = _select_overall_top_features(mutual_info_matrix, otu_names, args.top_k)
    mi_indices = [otu_names.index(name) for name in mi_features]
    _plot_heatmap(
        args.outdir,
        "mutual_info_heatmap.png",
        "Mutual information per latent dimension",
        mutual_info_matrix[:, mi_indices],
        latent_names,
        mi_features,
    )

    spearman_features = _select_overall_top_features(
        np.abs(spearman_matrix.T), otu_names, args.top_k
    )
    spearman_indices = [otu_names.index(name) for name in spearman_features]
    _plot_heatmap(
        args.outdir,
        "spearman_heatmap.png",
        "Spearman correlation per latent dimension",
        spearman_matrix.T[:, spearman_indices],
        latent_names,
        spearman_features,
    )

    _write_dataframe(
        os.path.join(args.outdir, "embeddings.tsv"),
        embeddings,
        index=sample_names,
        columns=latent_names,
    )

    _write_dataframe(
        os.path.join(args.outdir, "shap_mean_abs.tsv"),
        shap_mean_abs,
        index=latent_names,
        columns=otu_names,
    )

    _write_dataframe(
        os.path.join(args.outdir, "shap_mean.tsv"),
        shap_mean,
        index=latent_names,
        columns=otu_names,
    )

    _write_dataframe(
        os.path.join(args.outdir, "mutual_info.tsv"),
        mutual_info_matrix,
        index=latent_names,
        columns=otu_names,
    )

    _write_dataframe(
        os.path.join(args.outdir, "spearman.tsv"),
        spearman_matrix.T,
        index=latent_names,
        columns=otu_names,
    )

    if args.save_sample_shap:
        np.savez_compressed(
            os.path.join(args.outdir, "shap_values.npz"),
            shap_values=shap_values,
            sample_index=explain_idx,
            otu_names=np.asarray(otu_names),
            latent_names=np.asarray(latent_names),
            expected_value=expected,
        )

    summary = _summarize_top_features(
        otu_names,
        latent_names,
        shap_mean_abs,
        shap_mean,
        spearman_matrix,
        mutual_info_matrix,
        args.top_k,
    )
    summary.to_csv(os.path.join(args.outdir, "otu_latent_summary.tsv"), sep="\t", index=False)

    bg_set = set(bg_idx.tolist())
    explain_set = set(explain_idx.tolist())
    metadata = pd.DataFrame({"sample": sample_names})
    metadata["in_background"] = metadata.index.isin(bg_set).astype(int)
    metadata["in_explained"] = metadata.index.isin(explain_set).astype(int)
    metadata.to_csv(os.path.join(args.outdir, "sample_selection.tsv"), sep="\t", index=False)

    expected_vec = _normalize_expected_values(expected, latent_dim)
    expected_df = pd.DataFrame({"latent": latent_names, "expected_value": expected_vec})
    expected_df.to_csv(os.path.join(args.outdir, "shap_expected_value.tsv"), sep="\t", index=False)

    print(
        "Saved interpretation artifacts to",
        args.outdir,
        "(embeddings, SHAP summaries, mutual information, and Spearman analyses)",
    )


if __name__ == "__main__":
    main()
