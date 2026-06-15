"""CLI entry point for metadata classification using VAE embeddings.

Evaluates an XGBoost classifier on latent embeddings via repeated stratified
k-fold cross-validation.
"""

from __future__ import annotations

import argparse
import os

from biomevae.classify import (
    DEFAULT_EVAL_SEEDS,
    evaluate_embedding_classification,
    print_classification_summary,
    save_classification_results,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-classify",
        description=(
            "Classify sample metadata (e.g. disease status) using learned "
            "VAE embeddings. Evaluates multiple classifiers via repeated "
            "stratified k-fold cross-validation."
        ),
    )
    ap.add_argument(
        "--embeddings", required=True,
        help="Path to embeddings.tsv produced by biomevae-embed.",
    )
    ap.add_argument(
        "--metadata", required=True,
        help="Path to sample_metadata.tsv.",
    )
    ap.add_argument(
        "--label", default="disease",
        help="Metadata column to classify (default: disease).",
    )
    ap.add_argument(
        "--outdir", required=True,
        help="Output directory for classification results.",
    )
    ap.add_argument(
        "--prefix", default="",
        help="Output file prefix.",
    )
    ap.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of CV folds per seed (default: 5).",
    )
    ap.add_argument(
        "--n-repeats", type=int, default=10,
        help="Number of CV repetitions per seed (default: 10).",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS),
        help=(
            "Random seeds to repeat the evaluation over for reproducibility "
            "(default: %(default)s). Results are pooled across all seeds."
        ),
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help=(
            "DEPRECATED: legacy single-seed alias. If provided, overrides "
            "--seeds with a single-seed evaluation."
        ),
    )
    ap.add_argument(
        "--classifiers", nargs="+", default=None,
        choices=["XGBoost"],
        help="Subset of classifiers to evaluate (default: all).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    seeds = [args.seed] if args.seed is not None else list(args.seeds)

    print(f"[classify] Embeddings: {args.embeddings}")
    print(f"[classify] Metadata:   {args.metadata}")
    print(f"[classify] Label:      {args.label}")
    print(f"[classify] Seeds:      {seeds}")

    results = evaluate_embedding_classification(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        label_col=args.label,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seeds=seeds,
        classifier_names=args.classifiers,
    )

    print_classification_summary(results)

    path = save_classification_results(results, args.outdir, prefix=args.prefix)
    print(f"Results saved to: {path}")

    # Print detailed report for best classifier
    best_name = max(results, key=lambda k: results[k].balanced_accuracy)
    best = results[best_name]
    print(f"\nBest classifier: {best_name}")
    print(f"  Balanced accuracy: {best.balanced_accuracy:.4f}")
    print(f"  F1 macro:          {best.f1_macro:.4f}")
    if best.auroc is not None:
        print(f"  AUROC:             {best.auroc:.4f}")
    print(f"\n{best.classification_report}")


if __name__ == "__main__":
    main()
