"""Command line entry points exposed by :mod:`biomevae`.

The modules are imported lazily so that importing :mod:`biomevae.cli` does not
pull in heavy dependencies such as PyTorch until a specific subcommand is
referenced.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__all__ = [
    "vae_train",
    "vae_train_vanilla",
    "vae_train_taxaware",
    "vae_train_hyper",
    "vae_train_hyper_tax",
    "vae_train_graph",
    "vae_train_treeprior",
    "vae_train_fuse",
    "vae_train_flowxformer",
    "vae_embed",
    "vae_test",
    "vae_train_hgvae_zi",
    "vae_train_tree_dtm_vae",
    "vae_train_diva_tree_dtm_vae",
    "vae_train_phylodiva_tree_dtm_vae",
    "vae_train_philrvae",
    "vae_train_hyp_philrvae",
    "vae_train_diva_hyp_philrvae",
    "vae_train_phylodiva_hyp_philrvae",
    "vae_train_diva_betavae",
    "loso_prepare",
    "loso_classify",
    "loso_diagnostic",
    "nmf_baseline",
    "compare_to_nmf",
    "all_methods",
    "benchmark_figure",
    "benchmark_slides",
    "benchmark_figures_enterosignatures",
    "vae_interpret",
    "interpret_compare",
    "plot_training_curves",
    "recon_scatter",
    "recon_violin",
    "hierarchy_figure",
    "pairwise_table",
]

_lazy_modules: Dict[str, str] = {
    "vae_train": "vae_train",
    "vae_train_vanilla": "vae_train_vanilla",
    "vae_train_taxaware": "vae_train_taxaware",
    "vae_train_hyper": "vae_train_hyper",
    "vae_train_hyper_tax": "vae_train_hyper_tax",
    "vae_train_graph": "vae_train_graph",
    "vae_train_treeprior": "vae_train_treeprior",
    "vae_train_fuse": "vae_train_fuse",
    "vae_train_flowxformer": "vae_train_flowxformer",
    "vae_train_hgvae_zi": "vae_train_hgvae_zi",
    "vae_train_tree_dtm_vae": "vae_train_tree_dtm_vae",
    "vae_train_diva_tree_dtm_vae": "vae_train_diva_tree_dtm_vae",
    "vae_train_phylodiva_tree_dtm_vae": "vae_train_phylodiva_tree_dtm_vae",
    "vae_train_philrvae": "vae_train_philrvae",
    "vae_train_hyp_philrvae": "vae_train_hyp_philrvae",
    "vae_train_diva_hyp_philrvae": "vae_train_diva_hyp_philrvae",
    "vae_train_phylodiva_hyp_philrvae": "vae_train_phylodiva_hyp_philrvae",
    "vae_train_diva_betavae": "vae_train_diva_betavae",
    "loso_prepare": "loso_prepare",
    "loso_classify": "loso_classify",
    "loso_diagnostic": "loso_diagnostic",
    "vae_embed": "vae_embed",
    "vae_test": "vae_test",
    "nmf_baseline": "nmf_baseline",
    "compare_to_nmf": "compare_to_nmf",
    "all_methods": "all_methods",
    "benchmark_figure": "benchmark_figure",
    "benchmark_slides": "benchmark_slides",
    "benchmark_figures_enterosignatures": "benchmark_figures_enterosignatures",
    "vae_interpret": "vae_interpret",
    "interpret_compare": "interpret_compare",
    "plot_training_curves": "plot_training_curves",
    "recon_scatter": "recon_scatter",
    "recon_violin": "recon_violin",
    "hierarchy_figure": "hierarchy_figure",
    "pairwise_table": "pairwise_table",
}


def __getattr__(name: str) -> Any:
    if name in _lazy_modules:
        module = import_module(f".{_lazy_modules[name]}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
