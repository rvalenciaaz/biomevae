from __future__ import annotations

import argparse
from typing import Any, Dict

from biomevae.classify import DEFAULT_EVAL_SEEDS
from biomevae.reconstruction import (
    cross_validate_nmf_multi_seed,
    load_counts,
    merge_cross_val_results,
    select_nmf_rank,
)

from ._recon_cli import dump_result, parse_assignments, parse_int_list, result_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("biomevae-nmf")
    parser.add_argument("--input", required=True, help="Path to the counts matrix (TSV/CSV)")
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="Number of NMF components (ignored when --rank-candidates is provided)",
    )
    parser.add_argument(
        "--rank-candidates",
        default=None,
        help="Comma-separated list or range (e.g. 4,8,16 or 2-10) for NMF rank selection",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of Alex Williams CV splits")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of counts allocated to training in each split",
    )
    parser.add_argument(
        "--log1p",
        dest="log1p",
        action="store_true",
        help="Apply log1p transform before fitting",
    )
    parser.add_argument(
        "--no-log1p",
        dest="log1p",
        action="store_false",
        help="Disable log1p transform",
    )
    parser.set_defaults(log1p=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVAL_SEEDS),
        help=(
            "Random seeds to repeat the evaluation over (default: %(default)s). "
            "Fold metrics from every seed are pooled for reproducibility."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "DEPRECATED: legacy single-seed alias. If provided, overrides "
            "--seeds with a single-seed evaluation."
        ),
    )
    parser.add_argument(
        "--nmf-kw",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional keyword argument for sklearn.decomposition.NMF",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON summary (always printed to stdout)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    X = load_counts(args.input, log1p=False)
    nmf_kwargs: Dict[str, Any] = parse_assignments(args.nmf_kw)
    if args.components is None and not args.rank_candidates:
        raise SystemExit("Provide --components or --rank-candidates for the NMF baseline.")
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    print(f"[biomevae-nmf] Seeds: {seeds}")
    if args.rank_candidates:
        # ``select_nmf_rank`` chooses the best rank per seed individually and
        # then pools the selected ranks' fold metrics across seeds.
        candidates = parse_int_list(args.rank_candidates)
        per_seed = [
            select_nmf_rank(
                X,
                candidates=candidates,
                n_splits=args.splits,
                train_fraction=args.train_fraction,
                log1p=bool(args.log1p),
                nmf_kwargs=nmf_kwargs or None,
                random_state=s,
            )
            for s in seeds
        ]
        result = merge_cross_val_results(per_seed, seeds)
        # Preserve the per-seed rank selections for transparency.
        if result.metadata is None:
            result = type(result)(
                fold_metrics=result.fold_metrics,
                mean_metrics=result.mean_metrics,
                std_metrics=result.std_metrics,
                metadata={},
            )
        result.metadata["per_seed_selected_rank"] = {
            str(int(s)): r.metadata.get("selected_rank") if r.metadata else None
            for s, r in zip(seeds, per_seed)
        }
    else:
        result = cross_validate_nmf_multi_seed(
            X,
            n_components=args.components,
            n_splits=args.splits,
            train_fraction=args.train_fraction,
            log1p=bool(args.log1p),
            nmf_kwargs=nmf_kwargs or None,
            seeds=seeds,
        )
    payload = result_to_dict(result)
    dump_result(payload, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
