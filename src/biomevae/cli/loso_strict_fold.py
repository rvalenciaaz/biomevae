"""Build per-fold train/holdout splits for the strict LOSO pipeline.

Reads the full multi-study merged dataset (output of
``biomevae-loso-prepare``) and a single held-out study name, then writes
two parallel three-file bundles under ``--outdir``::

    <outdir>/train/sgb_table.tsv          (N-1 studies' columns)
    <outdir>/train/phyla.tsv              (== full phyla)
    <outdir>/train/sample_metadata.tsv    (N-1 studies' rows)
    <outdir>/holdout/sgb_table.tsv        (held-out study columns)
    <outdir>/holdout/phyla.tsv            (== full phyla)
    <outdir>/holdout/sample_metadata.tsv  (held-out study rows)
    <outdir>/fold_manifest.json

Both bundles share the *same feature ordering* as the full merged
dataset (columns / rows of ``sgb_table.tsv``), so a model trained on
``train/`` can be applied directly to ``holdout/sgb_table.tsv`` by the
strict-LOSO encode step without re-aligning features.

This is the only "data-side" piece needed for strict LOSO: every
encoder in the LOSO sweep can train on the smaller ``train/`` bundle
unchanged, because the loaders downstream
(:func:`biomevae.loso.load_merged`,
:func:`biomevae.cli._diva_common.build_diva_dataset`) read the standard
three-file layout.

Usage::

    biomevae-loso-strict-fold \\
        --merged   /scratch/biomevae_loso_runs/_merged_crc \\
        --held-out FengQ_2015 \\
        --outdir   /scratch/biomevae_loso_runs/loso_strict/crc/folds/FengQ_2015
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _read_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "sample_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sample_id"})
    return df


def _split_metadata(
    metadata: pd.DataFrame, held_out: str, study_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if study_col not in metadata.columns:
        raise SystemExit(
            f"biomevae-loso-strict-fold: metadata lacks '{study_col}' column."
        )
    holdout_mask = metadata[study_col].astype(str) == held_out
    if not holdout_mask.any():
        raise SystemExit(
            f"biomevae-loso-strict-fold: held-out study '{held_out}' not "
            f"present in {study_col} (values: "
            f"{sorted(metadata[study_col].astype(str).unique())[:10]} ...)."
        )
    train_mask = ~holdout_mask
    if not train_mask.any():
        raise SystemExit(
            f"biomevae-loso-strict-fold: removing '{held_out}' empties the "
            f"training set."
        )
    return metadata[train_mask].copy(), metadata[holdout_mask].copy()


def _slice_sgb_table(
    sgb_table: pd.DataFrame, sample_ids: List[str],
) -> pd.DataFrame:
    """Keep ``clade_name`` + ``NCBI_tax_id`` + the requested sample columns."""
    leading = ["clade_name", "NCBI_tax_id"]
    missing_lead = [c for c in leading if c not in sgb_table.columns]
    if missing_lead:
        raise SystemExit(
            f"biomevae-loso-strict-fold: merged sgb_table is missing required "
            f"column(s) {missing_lead}."
        )
    available = set(sgb_table.columns)
    kept = [s for s in sample_ids if s in available]
    missing = [s for s in sample_ids if s not in available]
    if missing:
        raise SystemExit(
            f"biomevae-loso-strict-fold: {len(missing)} metadata sample IDs "
            f"are absent from sgb_table.tsv (first few: {missing[:5]})."
        )
    return sgb_table[leading + kept]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-loso-strict-fold",
        description=(
            "Slice a merged multi-study dataset into a strict-LOSO fold: "
            "an N-1 training bundle and a held-out evaluation bundle, "
            "both with the full dataset's feature ordering preserved."
        ),
    )
    ap.add_argument(
        "--merged", required=True,
        help=(
            "Directory produced by biomevae-loso-prepare; must contain "
            "sgb_table.tsv, phyla.tsv and sample_metadata.tsv."
        ),
    )
    ap.add_argument(
        "--held-out", required=True,
        help="Study name to leave out (must appear in study_col).",
    )
    ap.add_argument(
        "--outdir", required=True,
        help="Output directory; train/ and holdout/ subfolders are created.",
    )
    ap.add_argument(
        "--study-col", default="study_name",
        help="Metadata column carrying the study label (default: study_name).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    merged = Path(args.merged)
    sgb_p = merged / "sgb_table.tsv"
    phyla_p = merged / "phyla.tsv"
    meta_p = merged / "sample_metadata.tsv"
    for p in (sgb_p, phyla_p, meta_p):
        if not p.exists():
            raise SystemExit(
                f"biomevae-loso-strict-fold: required file '{p}' not found."
            )

    outdir = Path(args.outdir)
    train_dir = outdir / "train"
    hold_dir = outdir / "holdout"
    train_dir.mkdir(parents=True, exist_ok=True)
    hold_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_metadata(meta_p)
    train_meta, hold_meta = _split_metadata(
        metadata, args.held_out, args.study_col,
    )

    sgb_table = pd.read_csv(sgb_p, sep="\t", dtype=str)
    train_sgb = _slice_sgb_table(sgb_table, train_meta["sample_id"].tolist())
    hold_sgb = _slice_sgb_table(sgb_table, hold_meta["sample_id"].tolist())

    train_sgb.to_csv(train_dir / "sgb_table.tsv", sep="\t", index=False)
    hold_sgb.to_csv(hold_dir / "sgb_table.tsv", sep="\t", index=False)
    # phyla is shared — copy verbatim to keep the existing three-file
    # layout invariants on both sides.
    shutil.copyfile(phyla_p, train_dir / "phyla.tsv")
    shutil.copyfile(phyla_p, hold_dir / "phyla.tsv")
    train_meta.to_csv(train_dir / "sample_metadata.tsv", sep="\t", index=False)
    hold_meta.to_csv(hold_dir / "sample_metadata.tsv", sep="\t", index=False)

    studies_in_train = sorted(
        train_meta[args.study_col].astype(str).unique().tolist()
    )
    manifest = {
        "merged_root": str(merged),
        "held_out_study": args.held_out,
        "study_col": args.study_col,
        "n_train_samples": int(train_meta.shape[0]),
        "n_holdout_samples": int(hold_meta.shape[0]),
        "n_features": int(train_sgb.shape[0]),
        "training_studies": studies_in_train,
    }
    with (outdir / "fold_manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        f"[loso-strict-fold] held_out={args.held_out} "
        f"train={manifest['n_train_samples']} samples / "
        f"{len(studies_in_train)} studies; "
        f"holdout={manifest['n_holdout_samples']} samples; "
        f"n_features={manifest['n_features']}"
    )
    print(f"[loso-strict-fold] wrote {train_dir}/ and {hold_dir}/")


if __name__ == "__main__":
    main()
