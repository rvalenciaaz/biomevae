"""CLI entry point for direct XGBoost classification from the SGB table.

Runs XGBoost on raw SGB abundances (no VAE embedding) as a classification
baseline, using repeated stratified k-fold cross-validation.
"""

from __future__ import annotations

import argparse

from biomevae.classify import (
    DEFAULT_EVAL_SEEDS,
    evaluate_direct_classification,
    print_classification_summary,
    save_classification_results,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-classify-baseline",
        description=(
            "Baseline classifier: XGBoost directly on the SGB abundance "
            "table (no dimensionality reduction). Evaluates via repeated "
            "stratified k-fold cross-validation."
        ),
    )
    ap.add_argument(
        "--input", required=True,
        help="Path to sgb_table.tsv (features x samples).",
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
        "--log1p", action="store_true", default=True,
        help="Apply log1p transform to abundances (default: True).",
    )
    ap.add_argument(
        "--no-log1p", action="store_false", dest="log1p",
        help="Disable log1p transform.",
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
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    seeds = [args.seed] if args.seed is not None else list(args.seeds)

    print(f"[classify-baseline] Input:    {args.input}")
    print(f"[classify-baseline] Metadata: {args.metadata}")
    print(f"[classify-baseline] Label:    {args.label}")
    print(f"[classify-baseline] log1p:    {args.log1p}")
    print(f"[classify-baseline] Seeds:    {seeds}")

    results = evaluate_direct_classification(
        sgb_table_path=args.input,
        metadata_path=args.metadata,
        label_col=args.label,
        log1p=args.log1p,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seeds=seeds,
    )

    print_classification_summary(results)

    path = save_classification_results(
        results, args.outdir, prefix="xgboost_baseline",
    )
    print(f"Results saved to: {path}")

    best_name = max(results, key=lambda k: results[k].balanced_accuracy)
    best = results[best_name]
    print(f"\nBaseline classifier: {best_name}")
    print(f"  Balanced accuracy: {best.balanced_accuracy:.4f}")
    print(f"  F1 macro:          {best.f1_macro:.4f}")
    if best.auroc is not None:
        print(f"  AUROC:             {best.auroc:.4f}")
    print(f"\n{best.classification_report}")


if __name__ == "__main__":
    main()
