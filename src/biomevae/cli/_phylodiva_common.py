"""Shared CLI / training helpers for the three PhyloDIVA backbones.

The three ``biomevae-train-phylodiva-*`` CLIs differ only in the
backbone-specific dataset prep, model class and reconstruction NLL.
Everything else — the GRL/BM/CORAL hyperparameters, the
``extra_loss_fn`` callback that feeds them to ``diva_train_loop``, the
DANN sigmoid schedule for ``lambda_GR`` — is identical and lives here.
"""
from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch

from biomevae.models.phylo_da import dann_lambda_schedule


__all__ = [
    "add_phylodiva_cli_args",
    "make_phylodiva_extra_loss_fn",
    "make_phylodiva_on_epoch_start",
    "phylodiva_config_dict",
]


def add_phylodiva_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register the PhyloDIVA-specific knobs (GRL / BM / CORAL).

    Defaults are taken from the literature:
      * ``lambda_gr_max=0.5``, ``gamma=10`` — Ganin et al. 2015 / 2016.
      * ``lambda_coral=1.0`` — Sun & Saenko 2016.
      * ``lambda_bm=1e-2`` — moderate Felsenstein-1985 BM weight.
    Optuna can override any of these via ``--optuna-config``.
    """
    parser.add_argument(
        "--lambda-gr-max", type=float, default=0.5,
        help=(
            "Maximum gradient-reversal coefficient for the hierarchical "
            "clade critic (DANN sigmoid schedule)."
        ),
    )
    parser.add_argument(
        "--lambda-bm", type=float, default=1e-2,
        help="Weight on the Brownian-motion smoothness penalty.",
    )
    parser.add_argument(
        "--lambda-coral", type=float, default=1.0,
        help="Weight on the per-study CORAL penalty on z_x.",
    )
    parser.add_argument(
        "--lambda-critic", type=float, default=1.0,
        help=(
            "Loss-side weight on the (already gradient-reversed) study "
            "critic CE.  Independent of --lambda-gr-max which controls "
            "the encoder-side sign-flip magnitude."
        ),
    )
    parser.add_argument(
        "--critic-hidden", type=int, default=64,
        help="Hidden width of the study critic's MLP head.",
    )
    parser.add_argument(
        "--study-balanced", action="store_true",
        help=(
            "Use a study-balanced batch sampler (>=2 studies per "
            "batch).  Recommended for the CORAL term."
        ),
    )
    parser.add_argument(
        "--gamma-gr", type=float, default=10.0,
        help="Gain on the DANN sigmoid schedule for lambda_GR.",
    )


def make_phylodiva_on_epoch_start(model: torch.nn.Module, args: argparse.Namespace):
    """Return a callback that sets the GRL coefficient each epoch."""
    def _cb(epoch_t: float) -> None:
        lam = dann_lambda_schedule(
            epoch_t, args.lambda_gr_max, gamma=args.gamma_gr,
        )
        # Each PhyloDIVA wrapper exposes ``model.critic.set_lambda``.
        model.critic.set_lambda(lam)
    return _cb


def make_phylodiva_extra_loss_fn(
    model: torch.nn.Module, args: argparse.Namespace,
) -> Callable[..., Dict[str, torch.Tensor]]:
    """Wrap ``model.extra_losses`` into the ``extra_loss_fn`` signature
    expected by ``diva_train_loop``.

    The callback ignores the ``batch`` and ``epoch_t`` arguments — every
    quantity it needs is already in ``out`` (raw counts, domain, z_x,
    edge_logits / coords_hat).  ``epoch_t`` controls the GRL schedule
    via ``on_epoch_start`` instead, so the per-step callback stays
    state-free.
    """
    def _cb(out: Dict[str, torch.Tensor], batch, epoch_t: float):
        del batch, epoch_t
        return model.extra_losses(
            out,
            lambda_bm=float(args.lambda_bm),
            lambda_coral=float(args.lambda_coral),
            lambda_critic=float(args.lambda_critic),
        )
    return _cb


def phylodiva_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Subset of args persisted into ``config.json`` for reproducibility."""
    return {
        "lambda_gr_max": float(args.lambda_gr_max),
        "lambda_bm": float(args.lambda_bm),
        "lambda_coral": float(args.lambda_coral),
        "lambda_critic": float(args.lambda_critic),
        "critic_hidden": int(args.critic_hidden),
        "gamma_gr": float(args.gamma_gr),
        "study_balanced": bool(getattr(args, "study_balanced", False)),
    }
