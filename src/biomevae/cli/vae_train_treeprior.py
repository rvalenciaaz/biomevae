from __future__ import annotations

import argparse
import os
import numpy as np

from biomevae.data import load_matrix
from biomevae.taxonomy import build_taxonomy_graph_from_taxonomy, load_feature_clades
from biomevae.trainers.train_loop import train_once

from .vae_train import _prepare_base_params, _run_optuna, build_parser as build_base_parser


def build_parser() -> argparse.ArgumentParser:
    ap = build_base_parser("biomevae-train-treeprior")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy table (TSV/CSV)")
    ap.add_argument(
        "--tax-graph-mode",
        choices=["unweighted", "branchlen"],
        default="unweighted",
        help="Edge weighting strategy for the taxonomy graph.",
    )
    ap.add_argument("--gnn", choices=["gcn"], default="gcn", help="Graph encoder architecture")
    ap.add_argument(
        "--gnn-hidden",
        nargs="+",
        type=int,
        default=[64],
        help="Hidden dimension(s) for the taxonomy graph encoder.",
    )
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--gnn-dropout", type=float, default=0.0)
    ap.add_argument("--prior", choices=["brownian"], default="brownian")
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--branch-reg", type=float, default=0.0)
    # TreeStructuredPriorVAE is particularly susceptible to posterior
    # collapse because its *prior* is learnable and per-sample — with no
    # free-bits floor the model can drive KL to 0 by matching prior to
    # posterior (the so-called "learnable-prior collapse").  Override
    # the base parser default (0.0) with a small positive floor.  Users
    # can still pass ``--free-bits 0`` to reproduce the legacy behaviour.
    ap.set_defaults(free_bits=0.02)
    return ap


def _resolve_gnn_hidden(args) -> list[int]:
    hidden = list(args.gnn_hidden)
    if len(hidden) == 1 and args.gnn_layers > 1:
        hidden = hidden * args.gnn_layers
    return hidden


def main() -> None:
    args = build_parser().parse_args()

    feature_clades = load_feature_clades(args.input)
    graph_spec = build_taxonomy_graph_from_taxonomy(
        feature_clades,
        args.taxonomy,
        mode=args.tax_graph_mode,
    )

    params = _prepare_base_params(args)
    params["model_type"] = "treeprior"
    params["feature_clades"] = feature_clades
    params["model_kwargs"] = {
        "graph_spec": graph_spec,
        "gnn_hidden": _resolve_gnn_hidden(args),
        "gnn_dropout": args.gnn_dropout,
        "graph_mode": args.tax_graph_mode,
        "prior_sigma": args.prior_sigma,
        "branch_reg": args.branch_reg,
        "prior_kind": args.prior,
        "gnn_type": args.gnn,
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
