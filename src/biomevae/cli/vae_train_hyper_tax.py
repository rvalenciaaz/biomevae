from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from biomevae.data import load_matrix
from biomevae.optuna_utils import build_trial_params, filter_trial_params, load_search_space
from biomevae.taxonomy import build_taxonomy_structures
from biomevae.trainers.train_loop import train_once


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-hyp-tax")
    ap.add_argument("--input", required=True)
    ap.add_argument("--taxonomy", required=True, help="Path to Phyla.csv (CSV/TSV).")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    ap.add_argument("--activation", type=str, default="leakyrelu", choices=["leakyrelu", "gelu", "relu"])
    ap.add_argument("--layer-norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--curvature", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    import torch as _torch  # lazy import for device detection

    ap.add_argument("--device", default="cuda" if _torch.cuda.is_available() else "cpu")

    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--standardize", action="store_true")

    ap.add_argument("--objective", type=str, default="beta", choices=["beta", "vanilla", "capacity"])
    ap.add_argument("--recon", type=str, default="mse", choices=["mse", "mae", "huber"])
    ap.add_argument("--huber-delta", type=float, default=1.0)

    ap.add_argument(
        "--kl-warmup",
        type=int,
        default=300,
        help="Absolute KL warmup length in epochs (default: 300, slow ramp).",
    )
    ap.add_argument(
        "--kl-warmup-frac",
        type=float,
        default=None,
        help=(
            "Optional: express KL warmup as a fraction of --epochs. When "
            "provided this OVERRIDES --kl-warmup."
        ),
    )
    ap.add_argument("--beta-max", type=float, default=0.05)
    ap.add_argument("--free-bits", type=float, default=0.0)

    ap.add_argument("--capacity-start", type=float, default=0.0)
    ap.add_argument("--capacity-end", type=float, default=None)
    ap.add_argument("--capacity-epochs", type=int, default=120)
    ap.add_argument("--capacity-gamma", type=float, default=1.0)

    ap.add_argument("--tax-loss-levels", nargs="+", default=["g", "f"], help="subset of k p c o f g s")
    ap.add_argument(
        "--tax-loss-weight",
        type=float,
        default=0.2,
        help="λ for hierarchical loss (per-level)",
    )
    ap.add_argument(
        "--tax-laplacian",
        type=float,
        default=0.0,
        help="γ for Laplacian smoothness on recon",
    )
    ap.add_argument(
        "--tax-lap-weights",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.25],
        help="Affinity weights for same-(species,genus,family) when building W",
    )

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


def _prepare_base_params(args) -> Dict[str, Any]:
    return {
        "device": args.device,
        "model_type": "hyperbolic",
        "model_kwargs": {"curvature": args.curvature},
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
        "objective": args.objective,
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
        "tax_levels": list(args.tax_loss_levels),
        "tax_loss_weight": args.tax_loss_weight,
        "tax_As": None,
        "lap_L": None,
        "lap_weight": args.tax_laplacian,
    }


def _attach_taxonomy(
    params: Dict[str, Any],
    tax_struct: Dict[str, Any],
    laplacian: Optional[np.ndarray],
) -> None:
    device = torch.device(params["device"])
    params["tax_As"] = {
        lvl: torch.tensor(A, dtype=torch.float32, device=device)
        for lvl, A in tax_struct["A_mats"].items()
    }
    if params.get("lap_weight", 0.0) > 0.0 and laplacian is not None:
        params["lap_L"] = torch.tensor(laplacian, dtype=torch.float32, device=device)
    else:
        params["lap_L"] = None


def _run_optuna(
    args,
    X: np.ndarray,
    sample_names: List[str],
    base_params: Dict[str, Any],
    tax_struct: Dict[str, Any],
) -> None:
    if args.optuna_trials <= 0:
        raise SystemExit("--optuna-trials must be a positive integer.")
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Optuna is not installed. Install with `pip install biomevae[optuna]`."
        ) from exc

    config = None
    if args.optuna_config:
        try:
            config = load_search_space(args.optuna_config)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    study_dir = os.path.join(args.outdir, "optuna_trials")
    os.makedirs(study_dir, exist_ok=True)

    X_raw = X.astype(np.float32, copy=True)
    X_log = np.log1p(X_raw).astype(np.float32)
    laplacian = tax_struct.get("L")

    def objective(trial):
        params = build_trial_params(trial, base_params, config)
        X_in = X_log if params["log1p"] else X_raw
        trial_outdir = os.path.join(study_dir, f"trial_{trial.number:04d}")
        seed = args.seed + trial.number
        record = filter_trial_params(params)
        trial.set_user_attr("params", record)
        trial.set_user_attr("full_params", copy.deepcopy(params))
        _attach_taxonomy(params, tax_struct, laplacian)
        res = train_once(
            X_in,
            sample_names,
            trial_outdir,
            params,
            seed=seed,
            verbose=False,
            return_model=False,
        )
        trial.set_user_attr("seed", seed)
        trial.set_user_attr("active_units", int(res.get("active_units", 0)))
        # Demote posterior-collapsed trials so Optuna's val_recon
        # minimisation can't select degenerate hyperparameters.
        from .vae_train import collapse_aware_score
        return collapse_aware_score(res, params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.optuna_trials)

    best_trial = study.best_trial
    best_params = copy.deepcopy(best_trial.user_attrs["full_params"])
    best_record = copy.deepcopy(best_trial.user_attrs["params"])
    best_seed = best_trial.user_attrs["seed"]
    params_for_training = copy.deepcopy(best_params)
    _attach_taxonomy(params_for_training, tax_struct, laplacian)
    X_final = X_log if params_for_training["log1p"] else X_raw
    res = train_once(
        X_final,
        sample_names,
        args.outdir,
        params_for_training,
        seed=best_seed,
        verbose=True,
        return_model=False,
    )
    _save_optuna_artifacts(args.outdir, best_record, study)
    print(f"\nOptuna best trial #{best_trial.number} | val={best_trial.value:.6f}")
    print(f"\nBest val loss: {res['best_val']:.6f}")


def main() -> None:
    args = build_parser().parse_args()
    X, sample_names = load_matrix(args.input, log1p=False)
    tax_struct = build_taxonomy_structures(
        input_path=args.input,
        taxonomy_path=args.taxonomy,
        levels=args.tax_loss_levels,
        lap_w=args.tax_lap_weights,
        verbose=True,
    )
    params = _prepare_base_params(args)

    if args.optuna:
        _run_optuna(args, X, sample_names, params, tax_struct)
        return

    os.makedirs(args.outdir, exist_ok=True)
    X_in = np.log1p(X).astype(np.float32) if args.log1p else X.astype(np.float32)
    laplacian = tax_struct.get("L")
    _attach_taxonomy(params, tax_struct, laplacian)
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
