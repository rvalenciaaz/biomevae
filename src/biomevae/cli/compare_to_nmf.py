from __future__ import annotations

import argparse
from typing import Dict, Mapping

from biomevae.classify import DEFAULT_EVAL_SEEDS
from biomevae.reconstruction import compare_with_nmf_multi_seed, load_counts
from biomevae.taxonomy import TAX_LEVELS_ALL, build_taxonomy_structures

from ._recon_cli import dump_result, load_json, parse_int_list, result_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("biomevae-comparetonmf")
    parser.add_argument("--input", required=True, help="Path to the counts matrix (TSV/CSV)")
    parser.add_argument(
        "--method-name",
        required=True,
        help="Identifier for the neural method being evaluated",
    )
    parser.add_argument(
        "--method-config",
        required=True,
        help="Path to a JSON file containing the training configuration for the method",
    )
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
        "--device",
        default=None,
        help="Optional device override for the neural method (e.g. cpu, cuda)",
    )
    parser.add_argument(
        "--taxonomy",
        default=None,
        help="Optional taxonomy table (TSV/CSV) used to compute hierarchy-aware metrics",
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
    vae_params = dict(load_json(args.method_config))
    if args.device:
        vae_params["device"] = args.device
    vae_params.setdefault("device", "cpu")
    taxonomy_eval = None
    if args.taxonomy:
        raw_levels = vae_params.get("tax_levels", [])
        if isinstance(raw_levels, str):
            levels = [tok for tok in raw_levels.replace(",", " ").split() if tok]
        elif isinstance(raw_levels, (list, tuple)):
            levels = [str(tok) for tok in raw_levels]
        else:
            levels = []
        if not levels:
            levels = list(TAX_LEVELS_ALL)
        tax_struct = build_taxonomy_structures(
            input_path=args.input,
            taxonomy_path=args.taxonomy,
            levels=sorted(set(levels)),
            lap_w=[0.0, 0.0, 0.0],
            verbose=False,
        )
        taxonomy_eval = {lvl: mat for lvl, mat in tax_struct["A_mats"].items()}
    if args.components is None and not args.rank_candidates:
        raise SystemExit("Provide --components or --rank-candidates for the NMF baseline.")
    nmf_rank_candidates = parse_int_list(args.rank_candidates) if args.rank_candidates else None
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    print(f"[biomevae-comparetonmf] Seeds: {seeds}")
    results = compare_with_nmf_multi_seed(
        X,
        method_name=args.method_name,
        vae_params=vae_params,
        nmf_components=args.components,
        nmf_rank_candidates=nmf_rank_candidates,
        n_splits=args.splits,
        train_fraction=args.train_fraction,
        seeds=seeds,
        taxonomy_eval=taxonomy_eval,
    )
    payload: Dict[str, Mapping[str, object]] = {
        name: result_to_dict(res) for name, res in results.items()
    }
    dump_result(payload, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
