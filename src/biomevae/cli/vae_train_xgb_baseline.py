"""Passthrough "trainer" for the LOSO XGBoost baseline.

XGBoost is a supervised classifier, not a representation learner, so it
does not fit the train-encoder-then-classify shape of every other LOSO
model.  Rather than introduce a parallel rule that bypasses the embed
step, this CLI plays the role of an unsupervised featurisation: it loads
the merged ``sgb_table.tsv``, optionally applies ``log1p``, and writes
the features as ``embeddings.tsv``.  The downstream
``biomevae-loso-classify`` rule then trains XGBoost on the train-fold raw
features and evaluates on the held-out study with the same
``StandardScaler`` + balanced class weights + 5 evaluation seeds as
every VAE row, so the numbers are directly comparable.

Usage::

    biomevae-train-xgb-baseline \\
        --input merged_sgb_table.tsv \\
        --outdir out/xgb-baseline

The ``--epochs`` / ``--optuna`` / ``--optuna-trials`` flags are accepted
and ignored so the global ``extra_args`` from the LOSO config flows
through unchanged.
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from biomevae.data import load_matrix


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-train-xgb-baseline",
        description=(
            "Passthrough featurisation for the LOSO XGBoost baseline.  "
            "Writes the (optionally log1p-transformed) merged SGB table as "
            "embeddings.tsv so the existing loso_classify rule trains "
            "XGBoost directly on raw features."
        ),
    )
    ap.add_argument(
        "--input", required=True,
        help="Merged sgb_table.tsv (output of biomevae-loso-prepare).",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--no-log1p", dest="log1p", action="store_false", default=True,
        help="Skip log1p transform on the SGB counts (default: applied).",
    )
    # Accept-and-ignore flags so the global LOSO ``extra_args`` (which is
    # tuned for the VAE trainers and includes ``--epochs --optuna ...``)
    # does not need a special case for the baselines.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--optuna", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=None)
    ap.add_argument("--optuna-config", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, sample_names = load_matrix(args.input, log1p=args.log1p)

    columns = [f"feat_{i}" for i in range(X.shape[1])]
    pd.DataFrame(X, index=sample_names, columns=columns).to_csv(
        outdir / "embeddings.tsv", sep="\t",
    )

    # Write a tiny model.pt so the Snakemake rule's declared output is
    # satisfied.  No predictive parameters live here — the actual XGBoost
    # classifier is fit per fold in biomevae-loso-classify.
    torch.save(
        {
            "model_type": "xgb-baseline",
            "log1p": bool(args.log1p),
            "n_features": int(X.shape[1]),
            "feature_names": columns,
        },
        outdir / "model.pt",
    )

    config = {
        "model_type": "xgb-baseline",
        "log1p": bool(args.log1p),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "argv": shlex.join(argv) if argv is not None else None,
        "note": (
            "Passthrough featurisation: the actual XGBoost fit happens per "
            "fold in biomevae-loso-classify on these features."
        ),
    }
    with (outdir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    print(
        f"[xgb-baseline] wrote {X.shape[0]} samples × {X.shape[1]} features "
        f"(log1p={args.log1p}) to {outdir / 'embeddings.tsv'}"
    )


if __name__ == "__main__":
    main()
