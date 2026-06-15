"""Control-anchor diagnostic on a trained LOSO model.

Computes pair-wise control-only **CORAL** (Frobenius covariance
distance) and **MMD²** (multi-bandwidth Gaussian kernel) on the latent
embeddings of every pair of studies in a merged dataset.  Both are
restricted to the *control* class so the metric reflects pure
batch/cohort drift unconfounded by disease prevalence.

Outputs:

* ``control_anchor_coral.tsv`` — long-format per-pair Frobenius distance.
* ``control_anchor_mmd.tsv``   — long-format per-pair MMD².
* ``control_anchor_summary.json`` — overall mean/median/max + the list
  of studies that contributed enough control samples to be included.

The defaults are tuned so a single invocation of this CLI produces the
"is DA needed?" diagnostic for any DIVA / non-DIVA embedding produced
by the rest of the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from biomevae.loso import control_anchor


def _load_embeddings(path: str):
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.values.astype(np.float32), list(df.index)


def _load_metadata(path: str):
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "sample_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sample_id"})
    return df.set_index("sample_id")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-loso-diagnostic",
        description=(
            "Pair-wise control-anchored CORAL and MMD diagnostics for "
            "trained biomevae embeddings."
        ),
    )
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument(
        "--latent-slice", default="full",
        choices=["full", "z_y", "z_x", "z_d"],
        help="Which latent factor to evaluate (DIVA only for z_*).",
    )
    ap.add_argument("--label", default="disease")
    ap.add_argument("--study-col", default="study_name")
    ap.add_argument("--control-value", default="healthy")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--outdir", required=True)
    return ap


def _resolve_embeddings_path(emb: str, slice_name: Optional[str]) -> str:
    if not slice_name or slice_name == "full":
        return emb
    sliced = Path(emb).with_name(f"embeddings_{slice_name}.tsv")
    if not sliced.exists():
        raise FileNotFoundError(
            f"loso-diagnostic: requested slice '{slice_name}' not found at "
            f"{sliced}."
        )
    return str(sliced)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    emb_path = _resolve_embeddings_path(args.embeddings, args.latent_slice)
    print(f"[loso-diagnostic] embeddings: {emb_path}")

    Z, sample_ids = _load_embeddings(emb_path)
    metadata = _load_metadata(args.metadata)
    anchor = control_anchor(
        Z, sample_ids, metadata,
        label_col=args.label,
        control_value=args.control_value,
        study_col=args.study_col,
        min_samples=int(args.min_samples),
    )
    paths = anchor.to_tsv(outdir)

    summary = {
        "n_studies_included": len(anchor.studies),
        "studies_included": anchor.studies,
        "coral": {
            "mean": float(anchor.coral_pairs["frobenius"].mean())
                if not anchor.coral_pairs.empty else None,
            "median": float(anchor.coral_pairs["frobenius"].median())
                if not anchor.coral_pairs.empty else None,
            "max": float(anchor.coral_pairs["frobenius"].max())
                if not anchor.coral_pairs.empty else None,
            "n_pairs": int(len(anchor.coral_pairs)),
        },
        "mmd": {
            "mean": float(anchor.mmd_pairs["mmd2"].mean())
                if not anchor.mmd_pairs.empty else None,
            "median": float(anchor.mmd_pairs["mmd2"].median())
                if not anchor.mmd_pairs.empty else None,
            "max": float(anchor.mmd_pairs["mmd2"].max())
                if not anchor.mmd_pairs.empty else None,
            "n_pairs": int(len(anchor.mmd_pairs)),
        },
        "config": {
            "latent_slice": args.latent_slice,
            "label_col": args.label,
            "control_value": args.control_value,
            "study_col": args.study_col,
            "min_samples": int(args.min_samples),
        },
        "files": paths,
    }
    summary_path = outdir / "control_anchor_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[loso-diagnostic] wrote {paths['coral']}")
    print(f"[loso-diagnostic] wrote {paths['mmd']}")
    print(f"[loso-diagnostic] wrote {summary_path}")


if __name__ == "__main__":
    main()
