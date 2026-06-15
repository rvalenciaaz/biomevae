"""Every ``biomevae-train-*`` CLI must expose ``--optuna``.

Regression guard for the LOSO and single-study pipelines: the workflow
forwards ``--optuna --optuna-trials N`` to every model's train CLI (see
``workflow/config/loso_strict_crc.yaml`` and ``workflow/Snakefile``), so a
backbone whose parser silently drops the flag breaks the whole run with
"unrecognized arguments".  This test imports each train CLI and asserts
that its ``build_parser`` registers the three Optuna flags consumed by
the shared Snakemake recipe.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("torch")  # most CLI modules pull torch in at import time


# Modules whose ``build_parser`` exposes the standard ``--optuna``
# trio.  CLI modules that don't define a ``build_parser`` (e.g. the
# dsvae / flowxformer / xgb modules) handle their parsers differently
# and are exercised by their dedicated tests; the focus here is the
# generic VAE / DIVA / PhyloDIVA / TAXI sweep used by the LOSO
# pipeline.
TRAIN_CLI_MODULES = [
    "biomevae.cli.vae_train",
    "biomevae.cli.vae_train_vanilla",
    "biomevae.cli.vae_train_taxaware",
    "biomevae.cli.vae_train_hyper",
    "biomevae.cli.vae_train_hyper_tax",
    "biomevae.cli.vae_train_graph",
    "biomevae.cli.vae_train_treeprior",
    "biomevae.cli.vae_train_fuse",
    "biomevae.cli.vae_train_hgvae_zi",
    "biomevae.cli.vae_train_tree_dtm_vae",
    "biomevae.cli.vae_train_philrvae",
    "biomevae.cli.vae_train_hyp_philrvae",
    "biomevae.cli.vae_train_diva_tree_dtm_vae",
    "biomevae.cli.vae_train_diva_hyp_philrvae",
    "biomevae.cli.vae_train_diva_betavae",
    "biomevae.cli.vae_train_phylodiva_tree_dtm_vae",
    "biomevae.cli.vae_train_phylodiva_hyp_philrvae",
    "biomevae.cli.vae_train_phylodiva_betavae",
    "biomevae.cli.vae_train_taxi_tree_dtm_vae",
    "biomevae.cli.vae_train_taxi_hyp_philrvae",
]


def _import_or_skip(module_name):
    """Import the CLI module, skipping on optional-dep ImportError.

    Several backbones depend on optional extras (geoopt for the
    hyperbolic family, torch_geometric for HGVAE-ZI).  Treat those as
    "skip" because the *user* may legitimately not have them installed —
    a missing extra is not a CLI-contract regression and shouldn't make
    the test suite red.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        msg = str(exc).lower()
        for hint in ("geoopt", "torch_geometric", "torch-geometric"):
            if hint in msg:
                pytest.skip(
                    f"optional dependency missing for {module_name}: {exc}"
                )
        raise


@pytest.mark.parametrize("module_name", TRAIN_CLI_MODULES)
def test_train_cli_registers_optuna_flags(module_name):
    mod = _import_or_skip(module_name)
    parser = mod.build_parser()
    dests = {a.dest for a in parser._actions}
    missing = {"optuna", "optuna_trials", "optuna_config"} - dests
    assert not missing, (
        f"{module_name}.build_parser() is missing Optuna flags: {sorted(missing)}. "
        "Add `biomevae.cli._diva_common.add_optuna_cli_args(parser)` or "
        "delegate to `biomevae.cli.vae_train.build_parser` so the LOSO "
        "Snakemake recipe can forward `--optuna --optuna-trials N`."
    )


@pytest.mark.parametrize("module_name", TRAIN_CLI_MODULES)
def test_train_cli_accepts_optuna_argv(module_name):
    """The full ``--optuna --optuna-trials 1`` triplet must parse without
    error.  Catches scripts whose ``main`` reads ``args.optuna`` but never
    registered the flag (``AttributeError`` at runtime), or whose parser
    rejects ``--optuna-config`` with a custom default.
    """
    mod = _import_or_skip(module_name)
    parser = mod.build_parser()

    required = []
    for action in parser._actions:
        if action.required and action.dest not in {"help"}:
            # Provide a dummy value of the right cardinality for every
            # required flag so ``parse_args`` doesn't bail on missing
            # inputs.  Use ``/dev/null`` because some CLIs validate that
            # the path *exists* lazily inside ``main``, not at parse time.
            required.append(action.option_strings[0])
            required.append("/dev/null")

    argv = required + ["--optuna", "--optuna-trials", "1"]
    args = parser.parse_args(argv)
    assert args.optuna is True
    assert args.optuna_trials == 1
    assert args.optuna_config is None
