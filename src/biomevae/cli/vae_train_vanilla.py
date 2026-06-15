"""Command-line entry point for training a traditional (β=1) VAE."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np

from biomevae.data import load_matrix
from biomevae.trainers.train_loop import train_once

from .vae_train import _prepare_base_params, _run_optuna, build_parser


def build_vanilla_parser() -> argparse.ArgumentParser:
    """Reuse the generic VAE parser but lock the objective to the vanilla setting."""
    parser = build_parser("biomevae-train-vanilla", fixed_objective="vanilla")
    parser.description = (
        "Train a standard variational autoencoder with a unit KL weight (β=1). "
        "This is equivalent to running `biomevae-train` with `--objective vanilla`, "
        "but exposed as a dedicated convenience entry point."
    )
    # Collapse-resistant defaults. Historically this entry point disabled
    # KL warmup and free-bits entirely, which drove posterior collapse on
    # compositional microbiome data (MetaCardis Vanilla VAE predicted only
    # the majority class). A small free-bits floor plus a 10%-of-run β
    # warmup gives the encoder time to learn structure before full KL
    # pressure kicks in, matching the TreeNB-VAE convention.
    parser.set_defaults(free_bits=0.02, kl_warmup_frac=0.1)
    return parser


def _run_single(
    X: np.ndarray,
    sample_names: List[str],
    outdir: str,
    params: Dict[str, Any],
    *,
    seed: int,
    log1p: bool,
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    X_in = np.log1p(X).astype(np.float32) if log1p else X.astype(np.float32)
    return train_once(
        X_in,
        sample_names,
        outdir,
        params,
        seed=seed,
        verbose=True,
        return_model=False,
    )


def main() -> None:
    args = build_vanilla_parser().parse_args()
    X, sample_names = load_matrix(args.input, log1p=False)
    params = _prepare_base_params(args, objective="vanilla")

    if args.optuna:
        _run_optuna(args, X, sample_names, params, objective_override="vanilla")
        return

    res = _run_single(
        X,
        sample_names,
        args.outdir,
        params,
        seed=args.seed,
        log1p=args.log1p,
    )
    print(f"\nBest val loss: {res['best_val']:.6f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

