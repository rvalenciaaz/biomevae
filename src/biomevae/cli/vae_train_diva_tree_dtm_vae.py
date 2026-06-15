"""Training script for DIVA-Tree-DTM-VAE.

Usage::

    biomevae-train-diva-tree-dtm \\
        --input merged_sgb_table.tsv \\
        --metadata merged_sample_metadata.tsv \\
        --taxonomy phyla.tsv \\
        --outdir out/diva-tree-dtm-vae

The merged inputs are produced by ``biomevae-loso-prepare`` (see
``biomevae.cli.loso_prepare``).  Internally the CLI:

* Aligns the merged SGB table to the taxonomy tree leaves and aggregates
  to all tree nodes (``X_nodes``).
* Encodes ``study_name`` and the user-chosen label column as integer
  domain / class labels.
* Trains :class:`biomevae.models.diva_treedtmvae.DIVATreeDTMVAE` with the
  Dirichlet-tree-multinomial likelihood by default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from biomevae.cli._diva_common import (
    DEFAULT_DIVA_SEARCH_SPACE,
    DIVADatasetTensors,
    add_optuna_cli_args,
    domain_class_encoders,
    encode_full_dataset,
    run_diva_optuna,
    save_diva_outputs,
    split_train_val,
)
from biomevae.losses import beta_schedule
from biomevae.loso import load_merged
from biomevae.models.diva_treedtmvae import DIVATreeDTMVAE
from biomevae.models.taxonomy_tree import (
    aggregate_leaf_matrix_to_nodes,
    build_taxonomy_graph_from_phyla_tsv,
)
from biomevae.models.tree_dtm_vae import build_tree_topology


LIKELIHOOD_CHOICES = ("dirichlet_tree_multinomial", "tree_multinomial", "dirichlet_tree")


# Default Optuna search space for the Tree-DTM DIVA backbone.  Builds on
# the shared DIVA defaults (latent_d/y/x, lr, dropout, alpha_y, alpha_d,
# beta_max, kl_warmup_frac, free_bits, batch_size) and adds the Dirichlet-
# tree concentration prior knob which materially affects collapse on
# sparse compositional data.
DEFAULT_DIVA_TREE_DTM_SEARCH_SPACE: Dict[str, Any] = {
    **DEFAULT_DIVA_SEARCH_SPACE,
    "init_concentration": {
        "method": "suggest_float", "low": 5.0, "high": 200.0, "log": True,
    },
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-diva-tree-dtm")
    ap.add_argument("--input", required=True, help="Merged SGB table (TSV)")
    ap.add_argument("--metadata", required=True, help="Merged sample metadata (TSV)")
    ap.add_argument("--taxonomy", required=True, help="Taxonomy table (TSV)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--study-col", default="study_name")
    ap.add_argument("--label-col", default="disease")

    ap.add_argument(
        "--data-kind", choices=("counts", "relative"), default="relative",
    )
    ap.add_argument(
        "--likelihood", choices=LIKELIHOOD_CHOICES, default="dirichlet_tree",
    )

    # Architecture
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent-d", type=int, default=4)
    ap.add_argument("--latent-y", type=int, default=8)
    ap.add_argument("--latent-x", type=int, default=8)
    ap.add_argument("--encoder-layers", type=int, default=2)
    ap.add_argument("--decoder-hidden", type=int, default=256)
    ap.add_argument("--decoder-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux-hidden", type=int, default=64)
    ap.add_argument("--encoder-pseudocount", type=float, default=0.5)
    ap.add_argument("--init-concentration", type=float, default=50.0)

    # Optimisation
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--beta-max", type=float, default=1.0)
    ap.add_argument("--kl-warmup-frac", type=float, default=0.25)
    ap.add_argument("--free-bits", type=float, default=0.02)
    ap.add_argument("--alpha-d", type=float, default=1.0)
    ap.add_argument("--alpha-y", type=float, default=10.0)
    ap.add_argument("--unlabelled-y-prior-weight", type=float, default=1.0)
    ap.add_argument("--concentration-l2", type=float, default=1e-4)

    # Misc
    ap.add_argument("--keep-prefixes", action="store_true")
    ap.add_argument("--taxonomy-has-header", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    add_optuna_cli_args(ap)
    return ap


def _build_dataset(
    args, *, eps: float = 1.0,
) -> Tuple[DIVADatasetTensors, torch.Tensor, list, object]:
    """Return DIVA tensors plus the per-sample node tensor and topology."""
    X_raw, sample_ids, feature_clades, metadata = load_merged(
        args.input, args.metadata, study_col=args.study_col,
    )

    taxg = build_taxonomy_graph_from_phyla_tsv(
        Path(args.taxonomy),
        keep_prefixes=bool(args.keep_prefixes),
        has_header=bool(args.taxonomy_has_header),
        on_duplicate_leaf="ignore_same",
    )
    topo = build_tree_topology(taxg)
    leaf_names = [taxg.node_names[nid] for nid in taxg.leaf_ids]

    name_to_col = {c: i for i, c in enumerate(feature_clades)}
    n_leaves = topo.n_leaves
    X_leaf = np.zeros((X_raw.shape[0], n_leaves), dtype=np.float32)
    missing = []
    for li, n in enumerate(leaf_names):
        col = name_to_col.get(n)
        if col is None:
            missing.append(n)
        else:
            X_leaf[:, li] = X_raw[:, col].astype(np.float32)
    if missing:
        print(
            f"[align] WARNING: {len(missing)} taxonomy leaves are missing from the "
            f"merged SGB table (filled with zeros). Example: {missing[:5]}"
        )

    if args.data_kind == "counts":
        X_leaf = np.rint(np.clip(X_leaf, 0.0, None)).astype(np.float32)
    else:
        totals = X_leaf.sum(axis=1, keepdims=True)
        keep = totals[:, 0] > 0
        X_leaf[keep] = X_leaf[keep] / totals[keep]

    X_nodes = aggregate_leaf_matrix_to_nodes(taxg, X_leaf).astype(np.float32)

    domain_idx, klass_idx, domain_classes, class_classes = domain_class_encoders(
        metadata, sample_ids,
        study_col=args.study_col, label_col=args.label_col,
    )

    ds = DIVADatasetTensors(
        x_log=torch.from_numpy(np.log(X_leaf + eps).astype(np.float32)),
        x_raw=torch.from_numpy(X_leaf),
        domain=torch.from_numpy(domain_idx),
        klass=torch.from_numpy(klass_idx),
        sample_ids=list(sample_ids),
        domain_classes=domain_classes,
        class_classes=class_classes,
        feature_clades=list(leaf_names),
    )
    return ds, torch.from_numpy(X_nodes), leaf_names, topo


def _epoch_pass(
    model: DIVATreeDTMVAE,
    loader,
    optimizer,
    *,
    train: bool,
    beta: float,
    args,
    device: torch.device,
    likelihood: str,
    validate_counts: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    totals: Dict[str, float] = {}
    n_total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_nodes, domain, klass in loader:
            x_nodes = x_nodes.to(device, non_blocking=True)
            domain = domain.to(device, non_blocking=True)
            klass = klass.to(device, non_blocking=True)
            loss, metrics = model.loss(
                x_nodes, domain, klass=klass,
                likelihood=likelihood,
                beta=beta,
                alpha_d=args.alpha_d,
                alpha_y=args.alpha_y,
                unlabelled_y_prior_weight=args.unlabelled_y_prior_weight,
                free_bits=args.free_bits,
                concentration_l2=args.concentration_l2,
                validate_counts=validate_counts,
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            bsz = x_nodes.size(0)
            n_total += bsz
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bsz
    return {k: v / max(1, n_total) for k, v in totals.items()}


def _train(
    args: argparse.Namespace,
    outdir: Path,
    *,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run one DIVA-Tree-DTM-VAE training pass.

    Returns ``{"best_val": float, "log_rows": list}`` so the generic
    ``run_diva_optuna`` helper can score trials and pick a winner.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds, X_nodes, leaf_names, topo = _build_dataset(args)
    device = torch.device(args.device)
    likelihood = args.likelihood
    validate_counts = likelihood != "dirichlet_tree"

    n_samples = X_nodes.size(0)
    train_idx, val_idx = split_train_val(n_samples, args.val_split, args.seed)

    train_ds = torch.utils.data.TensorDataset(
        X_nodes[train_idx], ds.domain[train_idx], ds.klass[train_idx],
    )
    val_ds = torch.utils.data.TensorDataset(
        X_nodes[val_idx], ds.domain[val_idx], ds.klass[val_idx],
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DIVATreeDTMVAE(
        n_domains=len(ds.domain_classes),
        n_classes=len(ds.class_classes),
        topo=topo,
        hidden=args.hidden,
        latent_d=args.latent_d, latent_y=args.latent_y, latent_x=args.latent_x,
        encoder_layers=args.encoder_layers,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        aux_hidden=args.aux_hidden,
        encoder_pseudocount=args.encoder_pseudocount,
        init_concentration=args.init_concentration,
        likelihood=likelihood,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    warmup = max(1, int(args.epochs * args.kl_warmup_frac))
    best_val = float("inf")
    no_improve = 0
    log_rows = []
    model_path = outdir / "model.pt"

    for ep in range(1, args.epochs + 1):
        beta = beta_schedule(ep, warmup, args.beta_max)
        tr = _epoch_pass(
            model, train_dl, optimizer, train=True, beta=beta,
            args=args, device=device, likelihood=likelihood,
            validate_counts=validate_counts,
        )
        va = _epoch_pass(
            model, val_dl, optimizer, train=False, beta=beta,
            args=args, device=device, likelihood=likelihood,
            validate_counts=validate_counts,
        )
        row = {"epoch": ep, "beta": float(beta)}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        row["train_recon"] = tr.get("reconstruction_nll", float("nan"))
        row["val_recon"] = va.get("reconstruction_nll", float("nan"))
        log_rows.append(row)
        val_nll = va.get("reconstruction_nll", float("inf"))
        if not np.isfinite(val_nll):
            print("Non-finite val loss; stopping.")
            break
        scheduler.step(val_nll)
        improved = val_nll + 1e-9 < best_val
        if improved:
            best_val = val_nll
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if args.early_stop > 0 and no_improve >= args.early_stop:
                print("Early stopping.")
                break
        if verbose:
            print(
                f"ep {ep:03d} | beta={beta:.3f} | val_recon={val_nll:.3f} "
                f"kl={va.get('kl_total', 0):.3f} ce_d={va.get('ce_d', 0):.3f} "
                f"ce_y={va.get('ce_y', 0):.3f}"
            )

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    cfg = {
        "model_type": "diva-tree-dtm-vae",
        "likelihood": likelihood,
        "data_kind": args.data_kind,
        "hidden": args.hidden,
        "latent_d": args.latent_d, "latent_y": args.latent_y, "latent_x": args.latent_x,
        "encoder_layers": args.encoder_layers,
        "decoder_hidden": args.decoder_hidden,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "aux_hidden": args.aux_hidden,
        "encoder_pseudocount": args.encoder_pseudocount,
        "init_concentration": args.init_concentration,
        "n_domains": len(ds.domain_classes),
        "n_classes": len(ds.class_classes),
        "domain_classes": ds.domain_classes,
        "class_classes": ds.class_classes,
        "feature_clades": leaf_names,
        "model_kwargs": {
            "keep_prefixes": bool(args.keep_prefixes),
            "taxonomy_has_header": bool(args.taxonomy_has_header),
        },
    }

    embeddings = encode_full_dataset(
        model=model,
        encode_fn=lambda x: model.encode(x),
        inputs=X_nodes,
        batch_size=128,
        device=device,
    )

    # Reconstruction: leaf_prob × per-sample library size for comparable count-space output.
    recon_parts = []
    leaf_totals = ds.x_raw.sum(dim=1, keepdim=True).clamp(min=1.0)
    with torch.no_grad():
        for start in range(0, X_nodes.size(0), 128):
            batch = X_nodes[start : start + 128].to(device)
            enc = model.encode(batch)
            lp = model.decode_parts(enc["mu_d"], enc["mu_y"], enc["mu_x"])["leaf_prob"]
            lib = leaf_totals[start : start + 128].to(device)
            recon_parts.append((lp * lib).cpu().numpy())
    recon = np.concatenate(recon_parts, axis=0) if recon_parts else None

    save_diva_outputs(
        outdir=outdir,
        sample_ids=ds.sample_ids,
        feature_clades=leaf_names,
        embeddings=embeddings,
        recon=recon,
        log_rows=log_rows,
        config=cfg,
    )
    if verbose:
        print(f"Best val_recon: {best_val:.4f}")
    return {"best_val": float(best_val), "log_rows": log_rows}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.optuna:
        run_diva_optuna(
            args,
            lambda a, outdir, *, verbose=True: _train(a, outdir, verbose=verbose),
            default_search_space=DEFAULT_DIVA_TREE_DTM_SEARCH_SPACE,
        )
        return
    _train(args, Path(args.outdir), verbose=True)


if __name__ == "__main__":
    main()
