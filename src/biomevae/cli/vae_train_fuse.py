from __future__ import annotations

import argparse
import os
import numpy as np

from biomevae.data import load_matrix
from biomevae.taxonomy import build_phylo_embeddings, load_feature_clades
from biomevae.trainers.train_loop import train_once

from .vae_train import _prepare_base_params, _run_optuna, build_parser as build_base_parser


def build_parser() -> argparse.ArgumentParser:
    ap = build_base_parser("biomevae-train-fuse")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy table (TSV/CSV)")
    ap.add_argument("--phylo-embed", choices=["pca"], default="pca")
    ap.add_argument("--phylo-embed-dim", type=int, default=32)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    feature_clades = load_feature_clades(args.input)
    phylo = build_phylo_embeddings(
        feature_clades,
        args.taxonomy,
        method=args.phylo_embed,
        dim=args.phylo_embed_dim,
    )

    params = _prepare_base_params(args)
    params["model_type"] = "phylo_fusion"
    params["feature_clades"] = feature_clades
    params["model_kwargs"] = {
        "phylo_embeddings": phylo.tolist(),
        "phylo_method": args.phylo_embed,
        "phylo_dim": int(min(args.phylo_embed_dim, phylo.shape[1])),
    }

    X, sample_names = load_matrix(args.input, log1p=False)

    if args.optuna:
        _run_optuna(args, X, sample_names, params)
        return

    os.makedirs(args.outdir, exist_ok=True)
    X_in = np.log1p(X).astype(np.float32) if args.log1p else X.astype(np.float32)
    res = train_once(
        X_in,
        sample_names,
        args.outdir,
        params,
        seed=args.seed,
        verbose=True,
        return_model=False,
    )
    print(f"\nBest val loss: {res['best_val']:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
