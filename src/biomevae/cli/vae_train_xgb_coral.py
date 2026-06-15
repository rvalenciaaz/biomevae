"""Per-study CORAL alignment featurisation for the LOSO XGBoost-DA row.

The DIVA analog at the feature level: rather than learn a latent that
factors out site-specific variance (DIVA's ``z_d``) and then classify on
the class-anchored slice, we apply per-study CORAL alignment (Sun &
Saenko 2016) to the raw features and let XGBoost classify on what's
left.  Each sample's transform depends only on its own study's mean and
covariance, so the held-out study in the LOSO split is aligned with its
own (unsupervised) features and no class label leaks across the boundary.

Usage::

    biomevae-train-xgb-coral \\
        --input merged_sgb_table.tsv \\
        --metadata merged_sample_metadata.tsv \\
        --outdir out/xgb-coral

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

from biomevae.loso import coral_align, load_merged


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-train-xgb-coral",
        description=(
            "Per-study CORAL alignment featurisation for the LOSO "
            "XGBoost-DA baseline.  Each study is whitened and re-coloured "
            "to a shared reference distribution, removing per-cohort mean "
            "and covariance fingerprints before XGBoost is trained downstream."
        ),
    )
    ap.add_argument(
        "--input", required=True,
        help="Merged sgb_table.tsv (output of biomevae-loso-prepare).",
    )
    ap.add_argument(
        "--metadata", required=True,
        help="Merged sample_metadata.tsv (must contain the study column).",
    )
    ap.add_argument(
        "--study-col", default="study_name",
        help="Metadata column carrying the study label (default: study_name).",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--no-log1p", dest="log1p", action="store_false", default=True,
        help="Skip log1p transform before alignment (default: applied).",
    )
    ap.add_argument(
        "--ridge", type=float, default=1e-3,
        help=(
            "Diagonal regularisation as a fraction of tr(Sigma)/p added "
            "to each per-study covariance before eigendecomposition.  "
            "Required because per-study sample counts are typically "
            "smaller than the feature count (default: 1e-3)."
        ),
    )
    ap.add_argument(
        "--reference", default="mean",
        choices=["mean", "identity", "largest"],
        help=(
            "Shared target distribution: 'mean' averages per-study "
            "(mu, Sigma); 'identity' whitens every study to N(0, I); "
            "'largest' aligns to the cohort with the most samples "
            "(default: mean)."
        ),
    )
    # Accept-and-ignore flags so the global LOSO ``extra_args`` flows
    # through unchanged.
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

    X_raw, sample_ids, feature_clades, metadata = load_merged(
        args.input, args.metadata, study_col=args.study_col,
    )
    if args.log1p:
        X = np.log1p(X_raw).astype(np.float32)
    else:
        X = X_raw.astype(np.float32)

    by_id = metadata.reindex(sample_ids)
    if args.study_col not in by_id.columns:
        raise SystemExit(
            f"biomevae-train-xgb-coral: metadata lacks '{args.study_col}'."
        )
    studies = by_id[args.study_col].astype(str).fillna("UNKNOWN").to_numpy()

    X_aligned, stats = coral_align(
        X, studies, ridge=args.ridge, reference=args.reference,
    )

    columns = [f"feat_{i}" for i in range(X_aligned.shape[1])]
    pd.DataFrame(X_aligned, index=sample_ids, columns=columns).to_csv(
        outdir / "embeddings.tsv", sep="\t",
    )

    torch.save(
        {
            "model_type": "xgb-coral",
            "log1p": bool(args.log1p),
            "n_features": int(X_aligned.shape[1]),
            "feature_names": columns,
            "feature_clades": list(feature_clades),
            "ridge": float(args.ridge),
            "reference": str(args.reference),
            "study_col": str(args.study_col),
            "studies": sorted({s for s in studies if s != "_reference"}),
        },
        outdir / "model.pt",
    )

    n_per_study = {
        s: int(d["n"]) for s, d in stats.items() if s != "_reference"
    }
    config = {
        "model_type": "xgb-coral",
        "log1p": bool(args.log1p),
        "ridge": float(args.ridge),
        "reference": str(args.reference),
        "study_col": str(args.study_col),
        "n_features": int(X_aligned.shape[1]),
        "n_samples": int(X_aligned.shape[0]),
        "samples_per_study": n_per_study,
        "argv": shlex.join(argv) if argv is not None else None,
        "note": (
            "Per-study CORAL alignment as preprocessing.  Each sample is "
            "whitened by its own study's covariance and re-coloured to a "
            "shared reference; the actual XGBoost fit happens per fold in "
            "biomevae-loso-classify on these aligned features."
        ),
    }
    with (outdir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    print(
        f"[xgb-coral] aligned {X_aligned.shape[0]} samples across "
        f"{len(n_per_study)} studies; reference='{args.reference}', "
        f"ridge={args.ridge}; wrote {outdir / 'embeddings.tsv'}"
    )


if __name__ == "__main__":
    main()
