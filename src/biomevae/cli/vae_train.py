from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List

import numpy as np

from biomevae.data import load_matrix
from biomevae.optuna_utils import build_trial_params, filter_trial_params, load_search_space
from biomevae.trainers.train_loop import train_once


def build_parser(
    prog: str = "biomevae-train",
    *,
    fixed_objective: str | None = None,
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    ap.add_argument("--activation", type=str, default="leakyrelu", choices=["leakyrelu", "gelu", "relu"])
    ap.add_argument("--layer-norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    import torch  # lazy import

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--standardize", action="store_true")

    if fixed_objective is None:
        ap.add_argument("--objective", type=str, default="beta", choices=["beta", "vanilla", "capacity"])
    else:
        ap.set_defaults(objective=fixed_objective)
    ap.add_argument("--recon", type=str, default="mse", choices=["mse", "mae", "huber"])
    ap.add_argument("--huber-delta", type=float, default=1.0)

    ap.add_argument(
        "--kl-warmup",
        type=int,
        default=300,
        help=(
            "Absolute KL warmup length in epochs. The default (300) is a "
            "deliberately slow schedule that keeps β small through short "
            "runs — this is ML-effective for β-VAEs and avoids posterior "
            "collapse. The trade-off is that the plotted ELBO stays "
            "non-stationary; use val_recon for convergence, not val_loss."
        ),
    )
    ap.add_argument(
        "--kl-warmup-frac",
        type=float,
        default=None,
        help=(
            "Optional: express KL warmup as a fraction of --epochs instead "
            "of an absolute count. When provided this OVERRIDES --kl-warmup. "
            "Matches the TreeNB-VAE / HGVAE-ZI convention (e.g. 0.25 = β "
            "saturates at 25%% of training)."
        ),
    )
    ap.add_argument("--beta-max", type=float, default=0.05)
    ap.add_argument("--free-bits", type=float, default=0.0)

    ap.add_argument("--capacity-start", type=float, default=0.0)
    ap.add_argument("--capacity-end", type=float, default=None)
    ap.add_argument("--capacity-epochs", type=int, default=120)
    ap.add_argument("--capacity-gamma", type=float, default=1.0)

    ap.add_argument(
        "--optuna",
        action="store_true",
        help="Run Optuna hyperparameter search instead of a single training run.",
    )
    ap.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to evaluate when --optuna is enabled.",
    )
    ap.add_argument(
        "--optuna-config",
        type=str,
        default=None,
        help="Optional JSON file defining custom Optuna search space overrides.",
    )
    return ap


def _save_optuna_artifacts(outdir: str, params: Dict[str, Any], study) -> None:
    os.makedirs(outdir, exist_ok=True)
    serializable = {k: v for k, v in params.items() if k not in ("tax_As", "lap_L")}
    with open(os.path.join(outdir, "optuna_best_params.json"), "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    try:
        df = study.trials_dataframe()
    except Exception:
        return
    df.to_csv(os.path.join(outdir, "optuna_trials.csv"), index=False)


def _prepare_base_params(args, objective: str | None = None) -> Dict[str, Any]:
    objective = objective if objective is not None else getattr(args, "objective", "beta")
    return {
        "device": args.device,
        "model_type": "euclid",
        "model_kwargs": {},
        "latent_dim": args.latent_dim,
        "hidden": list(args.hidden),
        "activation": args.activation,
        "layer_norm": args.layer_norm,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "log1p": args.log1p,
        "standardize": args.standardize,
        "val_split": args.val_split,
        "early_stop": args.early_stop,
        "objective": objective,
        "recon": args.recon,
        "huber_delta": args.huber_delta,
        "kl_warmup": args.kl_warmup,
        "kl_warmup_frac": args.kl_warmup_frac,
        "beta_max": args.beta_max,
        "free_bits": args.free_bits,
        "capacity_start": args.capacity_start,
        "capacity_end": args.capacity_end,
        "capacity_epochs": args.capacity_epochs,
        "capacity_gamma": args.capacity_gamma,
        "tax_levels": [],
        "tax_loss_weight": 0.0,
        "tax_As": None,
        "lap_L": None,
        "lap_weight": 0.0,
    }


def collapse_aware_score(res: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Optuna objective value combining val-recon with a posterior-collapse penalty.

    Optuna maximises likelihood of picking hyperparameters whose ``val_recon``
    is small — which is exactly minimised by a fully collapsed posterior
    (KL → 0, decoder reduced to the unconditional mean). The penalty term
    reshapes the search surface so that fully collapsed trials are demoted
    behind any trial that keeps at least one active latent unit, without
    interfering with the normal ordering among healthy trials.

    Penalty scale (0.1) is chosen on the order of the typical val_recon
    values observed in the benchmark (~1e-2), so a fully collapsed trial is
    pushed behind every legitimately-training trial. Shared across every
    CLI entry point whose Optuna search uses the generic ``train_once``
    return dict (``active_units``, ``latent_dim`` fields).
    """
    latent_dim = int(res.get("latent_dim", params.get("latent_dim", 1)) or 1)
    active_units = int(res.get("active_units", 0))
    collapse_frac = max(0.0, 1.0 - active_units / max(latent_dim, 1))
    return float(res["best_val"]) + 0.1 * collapse_frac


def _run_optuna(
    args,
    X: np.ndarray,
    sample_names: List[str],
    base_params: Dict[str, Any],
    *,
    objective_override: str | None = None,
) -> None:
    if args.optuna_trials <= 0:
        raise SystemExit("--optuna-trials must be a positive integer.")
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise SystemExit(
            "Optuna is not installed. Install with `pip install biomevae[optuna]`."
        ) from exc

    config = None
    if args.optuna_config:
        try:
            config = load_search_space(args.optuna_config)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    if objective_override is not None:
        override_cfg = {"objective": objective_override}
        if config is None:
            config = override_cfg
        else:
            config = dict(config)
            config.update(override_cfg)

    study_dir = os.path.join(args.outdir, "optuna_trials")
    os.makedirs(study_dir, exist_ok=True)

    X_raw = X.astype(np.float32, copy=True)
    X_log = np.log1p(X_raw).astype(np.float32)

    def objective(trial):
        params = build_trial_params(trial, base_params, config)
        X_in = X_log if params["log1p"] else X_raw
        trial_outdir = os.path.join(study_dir, f"trial_{trial.number:04d}")
        seed = args.seed + trial.number
        res = train_once(
            X_in,
            sample_names,
            trial_outdir,
            params,
            seed=seed,
            verbose=False,
            return_model=False,
        )
        record = filter_trial_params(params)
        trial.set_user_attr("params", record)
        trial.set_user_attr("full_params", copy.deepcopy(params))
        trial.set_user_attr("seed", seed)
        trial.set_user_attr("active_units", int(res.get("active_units", 0)))
        return collapse_aware_score(res, params)

    study = optuna.create_study(direction="minimize")
    # ``catch=(Exception,)`` keeps the search going when a single trial fails
    # (numerical instability, NaN loss, transient device error, …).  Without
    # this, one bad sample in a 100-trial sweep aborts the whole training and
    # leaves the LOSO pipeline with no embeddings.tsv, blocking every
    # downstream classify/diagnostic/aggregate job for this model.  Mirrors
    # the safety net already used by the DIVA helper in
    # ``biomevae.cli._diva_common.run_diva_optuna``.  ``KeyboardInterrupt``
    # derives from ``BaseException`` and is not caught.
    study.optimize(
        objective, n_trials=args.optuna_trials, catch=(Exception,),
    )

    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise SystemExit(
            f"Optuna: every trial failed before recording a val loss "
            f"({exc}). Inspect the per-trial logs under "
            f"{study_dir} for the original error."
        ) from exc
    best_params = copy.deepcopy(best_trial.user_attrs["full_params"])
    best_record = copy.deepcopy(best_trial.user_attrs["params"])
    best_seed = best_trial.user_attrs["seed"]
    X_final = X_log if best_params["log1p"] else X_raw
    res = train_once(
        X_final,
        sample_names,
        args.outdir,
        best_params,
        seed=best_seed,
        verbose=True,
        return_model=False,
    )
    _save_optuna_artifacts(args.outdir, best_record, study)
    print(f"\nOptuna best trial #{best_trial.number} | val={best_trial.value:.6f}")
    print(f"\nBest val loss: {res['best_val']:.6f}")


def main() -> None:
    args = build_parser(fixed_objective="beta").parse_args()
    X, sample_names = load_matrix(args.input, log1p=False)
    params = _prepare_base_params(args)

    if args.optuna:
        _run_optuna(args, X, sample_names, params, objective_override="beta")
        return

    os.makedirs(args.outdir, exist_ok=True)
    X_in = np.log1p(X).astype(np.float32) if args.log1p else X.astype(np.float32)
    res = train_once(
        X_in,
        sample_names,
        args.outdir,
        params,
        seed=args.seed,
        verbose=True,
        return_model=False,
    )
    print(f"\nBest val loss: {res['best_val']:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
