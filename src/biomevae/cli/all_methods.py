from __future__ import annotations

import argparse
import sys
from typing import Dict, Mapping

from biomevae.classify import DEFAULT_EVAL_SEEDS
from biomevae.reconstruction import compare_all_methods_multi_seed, load_counts
from biomevae.taxonomy import (
    TAX_LEVELS_ALL,
    build_phylo_embeddings,
    build_taxonomy_structures,
    load_feature_clades,
)

from ._recon_cli import dump_result, load_json, parse_int_list, result_to_dict

_TRAIN_ONCE_SUPPORTED_MODEL_TYPES = {
    "euclid",
    "hyperbolic",
    "graph_tax",
    "treeprior",
    "phylo_fusion",
    "philrvae",
    "tree-dtm-vae",
    "hyperbolic-philrvae",
    "dsvae",
}


def _parse_methods(specs: list[str]) -> Dict[str, Mapping[str, object]]:
    if not specs:
        raise SystemExit("At least one --method NAME=PATH argument is required.")
    methods: Dict[str, Mapping[str, object]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid method specification '{spec}'. Use NAME=PATH format.")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit("Method names cannot be empty.")
        if name in methods:
            raise SystemExit(f"Duplicate method name '{name}' provided.")
        methods[name] = dict(load_json(path.strip()))
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("biomevae-allcomp")
    parser.add_argument("--input", required=True, help="Path to the counts matrix (TSV/CSV)")
    parser.add_argument("--method", action="append", metavar="NAME=CONFIG", help="Neural method specification", required=True)
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
        help="Optional device override applied to all neural methods",
    )
    parser.add_argument(
        "--taxonomy",
        default=None,
        help="Optional taxonomy table (TSV/CSV) used to compute hierarchy-aware metrics",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print detailed progress updates while running comparisons",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON summary (always printed to stdout)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.verbose:
        print(f"biomevae-allcomp: loading counts from {args.input}")
    X = load_counts(args.input, log1p=False)
    if args.verbose:
        print("biomevae-allcomp: parsing method configurations")
    methods = _parse_methods(args.method)
    unsupported = {
        name: str(params.get("model_type", "euclid"))
        for name, params in methods.items()
        if str(params.get("model_type", "euclid")) not in _TRAIN_ONCE_SUPPORTED_MODEL_TYPES
    }
    if unsupported:
        skipped = ", ".join(
            f"{name}({model_type})" for name, model_type in sorted(unsupported.items())
        )
        supported = ", ".join(sorted(_TRAIN_ONCE_SUPPORTED_MODEL_TYPES))
        print(
            "biomevae-allcomp: skipping methods whose model_type is not "
            "implemented in cross_validate_vae: "
            f"{skipped}. Supported model_type values for this command: {supported}.",
            file=sys.stderr,
        )
        for name in unsupported:
            methods.pop(name, None)
    if not methods:
        print(
            "biomevae-allcomp: no supported methods remain after filtering; "
            "writing empty result payload.",
            file=sys.stderr,
        )
        dump_result({}, args.output)
        return
    if args.device:
        if args.verbose:
            print(f"biomevae-allcomp: overriding device for all methods -> {args.device}")
        for name, params in list(methods.items()):
            updated = dict(params)
            updated["device"] = args.device
            methods[name] = updated
    for name, params in list(methods.items()):
        if "device" not in params:
            updated = dict(params)
            updated["device"] = "cpu"
            methods[name] = updated

    _NEEDS_TAXONOMY_TYPES = {
        "phylo_fusion",
        "philrvae",
        "tree-dtm-vae",
        "hyperbolic-philrvae",
    }
    needs_taxonomy = [
        name for name, params in methods.items()
        if params.get("model_type") in _NEEDS_TAXONOMY_TYPES
    ]
    if needs_taxonomy and not args.taxonomy:
        raise SystemExit(
            "biomevae-allcomp: --taxonomy is required for "
            + ", ".join(sorted({str(methods[n].get("model_type")) for n in needs_taxonomy}))
            + " models."
        )

    # Inject taxonomy_path and input_path for tree-aware models that need
    # them during cross-validation training.  ``dsvae`` configs embed the
    # serialised ``tree_spec`` so they usually don't need the paths, but
    # we inject them as a fallback for hand-rolled configs.
    needs_nb_paths = [
        name for name, params in methods.items()
        if params.get("model_type") in (
            "philrvae", "tree-dtm-vae", "hyperbolic-philrvae", "dsvae",
        )
    ]
    for name in needs_nb_paths:
        params = dict(methods[name])
        params["taxonomy_path"] = args.taxonomy
        params["input_path"] = args.input
        methods[name] = params

    needs_phylo = [
        name for name, params in methods.items() if params.get("model_type") == "phylo_fusion"
    ]
    if needs_phylo:
        from biomevae.models.phylo_fusion import prepare_fusion_kwargs

        if args.verbose:
            joined = ", ".join(needs_phylo)
            print(f"biomevae-allcomp: preparing phylo_fusion embeddings for {joined}")
        feature_clades = load_feature_clades(args.input)
        for name in needs_phylo:
            params = dict(methods[name])
            kwargs = dict(params.get("model_kwargs", {}))
            if "phylo_embeddings" in kwargs:
                continue
            method = kwargs.get("phylo_method", "pca")
            dim = int(kwargs.get("phylo_dim", 32))
            if args.verbose:
                print(
                    "biomevae-allcomp: building phylo embeddings "
                    f"(method={method}, dim={dim}) for {name}"
                )
            phylo = build_phylo_embeddings(feature_clades, args.taxonomy, method=method, dim=dim)
            kwargs = prepare_fusion_kwargs({**kwargs, "phylo_embeddings": phylo})
            params["model_kwargs"] = kwargs
            methods[name] = params

    taxonomy_eval = None
    if args.taxonomy:
        if args.verbose:
            print(f"biomevae-allcomp: building taxonomy evaluation matrices from {args.taxonomy}")
        levels_needed: set[str] = set()
        for params in methods.values():
            raw_levels = params.get("tax_levels", [])
            if isinstance(raw_levels, str):
                tokens = [tok for tok in raw_levels.replace(",", " ").split() if tok]
            elif isinstance(raw_levels, (list, tuple)):
                tokens = [str(tok) for tok in raw_levels]
            else:
                tokens = []
            levels_needed.update(tokens)
        if not levels_needed:
            levels_needed.update(TAX_LEVELS_ALL)
        if levels_needed:
            tax_struct = build_taxonomy_structures(
                input_path=args.input,
                taxonomy_path=args.taxonomy,
                levels=sorted(levels_needed),
                lap_w=[0.0, 0.0, 0.0],
                verbose=args.verbose,
            )
            taxonomy_eval = {lvl: mat for lvl, mat in tax_struct["A_mats"].items()}

    if args.components is None and not args.rank_candidates:
        raise SystemExit("Provide --components or --rank-candidates for the NMF baseline.")
    nmf_rank_candidates = parse_int_list(args.rank_candidates) if args.rank_candidates else None
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if args.verbose:
        print(f"biomevae-allcomp: Seeds: {seeds}")
    results = compare_all_methods_multi_seed(
        X,
        methods=methods,
        nmf_components=args.components,
        nmf_rank_candidates=nmf_rank_candidates,
        n_splits=args.splits,
        train_fraction=args.train_fraction,
        seeds=seeds,
        taxonomy_eval=taxonomy_eval,
        verbose=args.verbose,
    )
    payload: Dict[str, Mapping[str, object]] = {
        name: result_to_dict(res) for name, res in results.items()
    }
    dump_result(payload, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
