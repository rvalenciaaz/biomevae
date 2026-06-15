"""Train the CAPDA-VAE and write stacking embeddings for the LOSO pipeline.

CAPDA-VAE (see :mod:`biomevae.models.capda_vae`) is a representation that
*augments* the raw ``log1p`` features with a leak-free, domain-aware
out-of-fold (OOF) invariant prediction, so it slots into the existing
train-encode-classify shape exactly like the ``xgb-baseline`` passthrough:

* this trainer writes ``embeddings.tsv`` = ``[log1p-species | OOF invariant
  probabilities]`` for the *input* (train-fold) samples, plus ``model.pt``
  (the final VAE) and ``config.json`` (everything needed to re-apply it);
* ``biomevae-loso-strict-encode`` applies the saved final VAE to the held-out
  cohort, emitting the same columns (final-VAE probabilities); and
* ``biomevae-loso-classify`` then fits XGBoost on the train rows and evaluates
  the held-out rows with the same StandardScaler + balanced weights + eval seeds
  as every other model, so the numbers are directly comparable.

In the **non-strict** pipeline the trainer is given the full N-study merge and
emits OOF embeddings for all samples (no separate encode step is needed — the
OOF probabilities for each study are already produced by a VAE that never saw
that study).

Usage::

    biomevae-train-capda-vae \\
        --input    merged_sgb_table.tsv \\
        --taxonomy phyla.tsv \\
        --metadata sample_metadata.tsv \\
        --outdir   out/capda-vae

The ``--epochs`` / ``--optuna`` / ``--optuna-trials`` / ``--optuna-config``
flags are accepted and ignored so the global LOSO ``extra_args`` flows through
unchanged; the VAE's own epoch budget is ``--vae-epochs`` (validated default).
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import pandas as pd
import torch

from biomevae.loso import load_merged
from biomevae.models.capda_vae import DEFAULTS, capda_fit, load_lineage_table


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-train-capda-vae",
        description=(
            "Train CAPDA-VAE (conditional alignment + CLR taxonomy bias) and "
            "write domain-aware OOF stacking embeddings for biomevae-loso-classify."
        ),
    )
    ap.add_argument("--input", required=True,
                    help="Merged sgb_table.tsv (biomevae-loso-prepare output).")
    ap.add_argument("--taxonomy", required=True, help="phyla.tsv for this group.")
    ap.add_argument("--metadata", required=True,
                    help="sample_metadata.tsv with study + disease columns.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-col", default="disease",
                    help="Metadata column with the class label (default: disease).")
    ap.add_argument("--study-col", default="study_name",
                    help="Metadata column with the study/domain (default: study_name).")
    ap.add_argument("--taxonomy-has-header", action="store_true",
                    help="Set if phyla.tsv has a header row (the merged pipeline "
                         "phyla.tsv is header-less; default assumes no header).")
    # CAPDA hyper-parameters (validated champion defaults).
    ap.add_argument("--vae-epochs", type=int, default=int(DEFAULTS["epochs"]),
                    help="Epochs per VAE fit (default: champion setting).")
    ap.add_argument("--latent", type=int, default=int(DEFAULTS["latent"]))
    ap.add_argument("--hidden", type=int, default=int(DEFAULTS["hidden"]))
    ap.add_argument("--gamma-cov", type=float, default=float(DEFAULTS["gamma_cov"]),
                    help="Weight of the conditional-covariance (CORAL) alignment.")
    ap.add_argument("--transform", default=str(DEFAULTS["transform"]),
                    choices=["clr", "log1p"],
                    help="Per-species compositional transform for the VAE input.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    # Accept-and-ignore flags so the global LOSO extra_args needs no special case.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--optuna", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=None)
    ap.add_argument("--optuna-config", type=str, default=None)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_raw, sample_ids, feat_clades, metadata = load_merged(
        args.input, args.metadata, study_col=args.study_col,
    )
    if args.label_col not in metadata.columns:
        raise SystemExit(
            f"biomevae-train-capda-vae: metadata lacks label column "
            f"'{args.label_col}'. Available: {list(metadata.columns)[:12]}..."
        )
    meta = metadata.reindex(sample_ids)
    y_raw = meta[args.label_col].to_numpy()
    study = meta[args.study_col].astype(str).to_numpy()
    taxonomy = load_lineage_table(
        args.taxonomy, has_header=bool(args.taxonomy_has_header))
    n_genus = int(taxonomy["g"].nunique()) if "g" in taxonomy.columns else 0
    print(f"[capda-vae] taxonomy: {len(taxonomy)} clades, {n_genus} genera "
          f"(has_header={bool(args.taxonomy_has_header)})")

    hp = dict(
        epochs=int(args.vae_epochs), latent=int(args.latent),
        hidden=int(args.hidden), gamma_cov=float(args.gamma_cov),
        transform=str(args.transform),
    )
    emb_df, state_dict, config = capda_fit(
        X_raw, sample_ids, feat_clades, study, y_raw, taxonomy,
        seed=int(args.seed), device=str(args.device), hp=hp,
    )

    emb_df.to_csv(outdir / "embeddings.tsv", sep="\t")
    torch.save(state_dict, outdir / "model.pt")
    config["argv"] = shlex.join(argv) if argv is not None else None
    config["seed"] = int(args.seed)
    config["label_col"] = args.label_col
    config["study_col"] = args.study_col
    config["taxonomy_has_header"] = bool(args.taxonomy_has_header)
    with (outdir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    print(
        f"[capda-vae] wrote {emb_df.shape[0]} samples × {emb_df.shape[1]} "
        f"features ({config['n_species']} log1p-species + {config['n_classes']} "
        f"OOF invariant-prob cols; classes={config['class_order']}) to "
        f"{outdir / 'embeddings.tsv'}"
    )


if __name__ == "__main__":
    main()
