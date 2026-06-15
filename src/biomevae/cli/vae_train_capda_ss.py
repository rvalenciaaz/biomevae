"""Train the single-study CAPDA-VAE for the standard / meta Snakemake pipeline.

This is the single-cohort sibling of :mod:`biomevae.cli.vae_train_capda_vae`
(the cross-cohort LOSO trainer).  The cross-study conditional alignment that
gives CAPDA its name is inert when every sample comes from one study, so the
single-study variant keeps the two parts that *do* transfer:

* the **multi-resolution taxonomy bias** — per-species CLR coordinates plus
  genus/family/order/phylum aggregates feed a supervised VAE; and
* **leak-free stacking** — class-head probabilities are produced *out-of-fold*
  with a stratified K-fold (the within-study analogue of LOSO's per-study
  holdout) so the columns handed downstream are not in-sample-leaky.

It matches the single-study ``train_model`` contract exactly::

    biomevae-train-capda-vae-ss \\
        --input    sgb_table.tsv \\
        --taxonomy phyla.tsv \\
        --metadata sample_metadata.tsv --label-col disease \\
        --outdir   out/capda-vae

and writes ``model.pt`` (the final VAE), ``config.json`` (everything needed to
re-apply it), and ``oof_embeddings.tsv`` (the leak-free ``[log1p-species |
OOF-prob]`` table that ``biomevae-embed`` passes through to ``biomevae-classify``).

The global ``extra_args`` (``--epochs`` / ``--optuna`` / ``--optuna-trials`` /
``--optuna-config``) are accepted and ignored so the shared single-study /meta
config flows through unchanged; the VAE's own budget is ``--vae-epochs``.
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from biomevae.data import load_matrix
from biomevae.models.capda_vae import (
    DEFAULTS,
    capda_fit_single_study,
    load_lineage_table,
)
from biomevae.taxonomy import load_feature_clades


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-train-capda-vae-ss",
        description=(
            "Train the single-study CAPDA-VAE (multi-resolution CLR taxonomy "
            "bias + leak-free stratified-K-fold OOF stacking) and write "
            "embeddings for biomevae-classify."
        ),
    )
    ap.add_argument("--input", required=True, help="sgb_table.tsv for one study.")
    ap.add_argument("--taxonomy", required=True, help="phyla.tsv for this study.")
    ap.add_argument("--metadata", required=True,
                    help="sample_metadata.tsv with the disease label column.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-col", default="disease",
                    help="Metadata column with the class label (default: disease).")
    ap.add_argument("--study-col", default="study_name",
                    help="Optional within-study sub-cohort column. When present "
                         "with >=2 levels it re-activates the conditional "
                         "alignment; otherwise it is ignored (default: study_name).")
    ap.add_argument("--taxonomy-has-header", action="store_true",
                    help="Set if phyla.tsv has a header row (default: header-less, "
                         "matching the extract-microbiome-data layout).")
    # CAPDA hyper-parameters (validated champion defaults).
    ap.add_argument("--vae-epochs", type=int, default=int(DEFAULTS["epochs"]),
                    help="Epochs per VAE fit (default: champion setting).")
    ap.add_argument("--latent", type=int, default=int(DEFAULTS["latent"]))
    ap.add_argument("--hidden", type=int, default=int(DEFAULTS["hidden"]))
    ap.add_argument("--gamma-cov", type=float, default=float(DEFAULTS["gamma_cov"]),
                    help="Weight of the conditional-covariance (CORAL) alignment "
                         "(only active with a >=2-level --study-col).")
    ap.add_argument("--transform", default=str(DEFAULTS["transform"]),
                    choices=["clr", "log1p"],
                    help="Per-species compositional transform for the VAE input.")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="Stratified K-fold splits for the leak-free OOF probabilities.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    # Accept-and-ignore flags so the global extra_args needs no special case.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--optuna", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=None)
    ap.add_argument("--optuna-config", type=str, default=None)
    return ap


def _read_metadata(metadata_path: str, sample_names: List[str]):
    """Load the metadata table indexed by the per-sample id column."""
    path = str(metadata_path)
    if path.endswith((".tsv", ".txt", ".tab")):
        meta = pd.read_csv(path, sep="\t", dtype=str)
    else:
        try:
            meta = pd.read_csv(path, sep="\t", dtype=str)
            if meta.shape[1] <= 1:
                meta = pd.read_csv(path, dtype=str)
        except Exception:
            meta = pd.read_csv(path, dtype=str)
    id_candidates = [
        "sample_id", "sample", "Sample", "Run", "run_accession",
        "sampleID", "ID", "id",
    ]
    id_col = next((c for c in id_candidates if c in meta.columns),
                  meta.columns[0])
    meta = meta.set_index(id_col)
    missing = [s for s in sample_names if s not in meta.index]
    if missing:
        raise SystemExit(
            f"biomevae-train-capda-vae-ss: {len(missing)} samples missing from "
            f"the metadata (first few: {missing[:5]})."
        )
    return meta.reindex(sample_names)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_raw, sample_ids = load_matrix(args.input, log1p=False)
    feat_clades = load_feature_clades(args.input)
    meta = _read_metadata(args.metadata, sample_ids)
    if args.label_col not in meta.columns:
        raise SystemExit(
            f"biomevae-train-capda-vae-ss: metadata lacks label column "
            f"'{args.label_col}'. Available: {list(meta.columns)[:12]}..."
        )
    y_raw = meta[args.label_col].to_numpy()

    # Within-study sub-cohort domain (optional). Only use it when the column is
    # present *and* has >=2 distinct levels among these samples — otherwise the
    # conditional alignment would be a no-op anyway.
    study = None
    if args.study_col in meta.columns:
        col = meta[args.study_col].astype(str)
        if col.nunique(dropna=True) >= 2:
            study = col.to_numpy()
            print(f"[capda-vae-ss] using within-study domain '{args.study_col}' "
                  f"({col.nunique()} levels) for conditional alignment")

    taxonomy = load_lineage_table(
        args.taxonomy, has_header=bool(args.taxonomy_has_header))
    n_genus = int(taxonomy["g"].nunique()) if "g" in taxonomy.columns else 0
    print(f"[capda-vae-ss] taxonomy: {len(taxonomy)} clades, {n_genus} genera "
          f"(has_header={bool(args.taxonomy_has_header)})")

    hp = dict(
        epochs=int(args.vae_epochs), latent=int(args.latent),
        hidden=int(args.hidden), gamma_cov=float(args.gamma_cov),
        transform=str(args.transform),
    )
    emb_df, state_dict, config = capda_fit_single_study(
        X_raw, sample_ids, feat_clades, y_raw, taxonomy,
        study=study, n_splits=int(args.n_splits), seed=int(args.seed),
        device=str(args.device), hp=hp,
    )

    # Persist the leak-free OOF embeddings so biomevae-embed can pass them
    # straight through to biomevae-classify (re-deriving them in-sample would
    # leak the label into the class-probability columns).
    emb_df.to_csv(outdir / "oof_embeddings.tsv", sep="\t")
    torch.save(state_dict, outdir / "model.pt")
    config["argv"] = shlex.join(argv) if argv is not None else None
    config["seed"] = int(args.seed)
    config["label_col"] = args.label_col
    config["study_col"] = args.study_col
    config["taxonomy_has_header"] = bool(args.taxonomy_has_header)
    with (outdir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    print(
        f"[capda-vae-ss] wrote {emb_df.shape[0]} samples x {emb_df.shape[1]} "
        f"features ({config['n_species']} log1p-species + {config['n_classes']} "
        f"OOF prob cols; classes={config['class_order']}; "
        f"OOF k={config['n_splits_effective']}) to {outdir / 'oof_embeddings.tsv'}"
    )


if __name__ == "__main__":
    main()
