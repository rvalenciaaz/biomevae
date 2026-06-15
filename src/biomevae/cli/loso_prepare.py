"""Build a merged multi-study dataset for the biomevae LOSO pipeline.

This is a thin CLI wrapper around :func:`biomevae.loso.merge_studies` that
takes a directory of per-study extracted folders and writes the standard
three-file layout (``sgb_table.tsv``, ``phyla.tsv``,
``sample_metadata.tsv``) into ``--outdir``.  The downstream LOSO rules
consume the resulting folder exactly the same way the existing pipeline
consumes a single-study extract.

Usage::

    biomevae-loso-prepare \\
        --data-root /scratch/extracted_studies \\
        --studies FengQ_2015,VogtmannE_2016,ZellerG_2014,YuJ_2015,WirbelJ_2018,\\
                  YachidaS_2019,ThomasAM_2018a,ThomasAM_2018b,ThomasAM_2019_c,\\
                  HanniganGD_2017,GuptaA_2019\\
        --outdir /scratch/extracted_studies/_merged_crc
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from biomevae.loso import merge_studies


def _parse_studies(arg: str) -> list[str]:
    if "," in arg:
        return [s.strip() for s in arg.split(",") if s.strip()]
    p = Path(arg)
    if p.exists() and p.is_file():
        return [
            line.strip()
            for line in p.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return [arg]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-loso-prepare",
        description=(
            "Concatenate per-study extracts into a single multi-study "
            "dataset suitable for leave-one-study-out training."
        ),
    )
    ap.add_argument(
        "--data-root", required=True,
        help="Directory containing per-study sub-folders.",
    )
    ap.add_argument(
        "--studies", required=True,
        help="Comma-separated list, or path to a one-per-line text file.",
    )
    ap.add_argument(
        "--outdir", required=True,
        help="Output directory; will hold the three merged TSVs.",
    )
    ap.add_argument(
        "--label-col", default="disease",
        help="Required metadata column to validate (default: disease).",
    )
    ap.add_argument(
        "--study-col", default="study_name",
        help="Output study-identifier column name (default: study_name).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    studies = _parse_studies(args.studies)
    if not studies:
        raise SystemExit("--studies resolved to an empty list.")

    print(f"[loso-prepare] merging {len(studies)} studies under {args.data_root}")
    merged = merge_studies(
        args.data_root, studies,
        require_columns=(args.label_col,),
        study_col=args.study_col,
    )
    paths = merged.write(args.outdir)

    manifest = {
        "studies": studies,
        "n_samples": int(merged.metadata.shape[0]),
        "n_features": len(merged.feature_clades),
        "n_studies": len(set(merged.metadata[args.study_col])),
        "label_col": args.label_col,
        "study_col": args.study_col,
        "files": paths,
    }
    manifest_path = Path(args.outdir) / "loso_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[loso-prepare] wrote {paths['sgb_table']}")
    print(f"[loso-prepare] wrote {paths['phyla']}")
    print(f"[loso-prepare] wrote {paths['sample_metadata']}")
    print(f"[loso-prepare] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
