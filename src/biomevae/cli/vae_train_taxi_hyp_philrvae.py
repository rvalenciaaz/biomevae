"""Training script for :class:`TAXIHyperbolicPhILRVAE`.

PhILR / hyperbolic counterpart of
:mod:`biomevae.cli.vae_train_taxi_tree_dtm_vae`.  Mirrors the PhyloDIVA
hyp-PhILR CLI but instantiates the TAXI variant — taxonomy-protected
conditional invariance on the residual ``z_rho`` latent only, with
class-conditional CORAL and an optional cross-covariance orthogonality
penalty between ``z_tau`` and ``z_rho``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data

from biomevae.cli._diva_common import (
    add_optuna_cli_args,
    encode_full_dataset,
    run_diva_optuna,
    save_diva_outputs,
    split_train_val,
)
from biomevae.cli.vae_train_diva_hyp_philrvae import (
    LIKELIHOOD_CHOICES,
    _build_dataset,
)
from biomevae.losses import beta_schedule
from biomevae.models.taxi_hyp_philrvae import TAXIHyperbolicPhILRVAE


# TAXI's λ knobs are unique to this variant; the latent-tau / latent-rho
# rename mirrors ``vae_train_taxi_tree_dtm_vae``.  The Euclidean DIVA
# default search space can't be reused unmodified because its keys
# (``latent_y`` / ``latent_x``) would be written back onto attributes
# that this parser doesn't expose under those names.
DEFAULT_TAXI_HYP_PHILRVAE_SEARCH_SPACE: Dict[str, Any] = {
    "latent_d": {"method": "suggest_categorical", "choices": [2, 4, 6, 8]},
    "latent_tau": {"method": "suggest_categorical", "choices": [4, 8, 12, 16]},
    "latent_rho": {"method": "suggest_categorical", "choices": [4, 8, 12, 16]},
    "lr": {"method": "suggest_float", "low": 1e-4, "high": 5e-3, "log": True},
    "dropout": {"method": "suggest_float", "low": 0.0, "high": 0.3},
    "alpha_y": {"method": "suggest_float", "low": 1.0, "high": 100.0, "log": True},
    "alpha_d": {"method": "suggest_float", "low": 0.5, "high": 5.0, "log": True},
    "beta_max": {"method": "suggest_float", "low": 0.02, "high": 1.0, "log": True},
    "kl_warmup_frac": {"method": "suggest_float", "low": 0.1, "high": 0.5},
    "free_bits": {"method": "suggest_float", "low": 0.0, "high": 0.05},
    "batch_size": {"method": "suggest_categorical", "choices": [32, 64, 128]},
    "curvature": {"method": "suggest_float", "low": 0.25, "high": 4.0, "log": True},
    "lambda_cond_critic": {
        "method": "suggest_float", "low": 1e-3, "high": 1.0, "log": True,
    },
    "lambda_cond_coral": {
        "method": "suggest_float", "low": 1e-3, "high": 1.0, "log": True,
    },
    "lambda_philr_smooth": {
        "method": "suggest_float", "low": 0.0, "high": 0.1,
    },
    "lambda_orth": {
        "method": "suggest_float", "low": 0.0, "high": 0.1,
    },
    "lambda_tau_aux": {
        "method": "suggest_float", "low": 0.0, "high": 1.0,
    },
    "init_concentration": {
        "method": "suggest_float", "low": 5.0, "high": 200.0, "log": True,
    },
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-taxi-hyp-philrvae")
    ap.add_argument("--input", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--study-col", default="study_name")
    ap.add_argument("--label-col", default="disease")

    ap.add_argument("--data-kind", choices=("counts", "relative"), default="relative")
    ap.add_argument("--likelihood", choices=LIKELIHOOD_CHOICES, default="philr_gaussian")

    ap.add_argument("--hidden", type=int, nargs="+", default=(256, 128))
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--latent-d", type=int, default=4)
    # TAXI uses tau/rho naming, but accept latent-y / latent-x aliases too so
    # users coming from DIVA / PhyloDIVA CLIs don't trip on the rename.
    ap.add_argument("--latent-tau", "--latent-y", dest="latent_tau", type=int, default=8)
    ap.add_argument("--latent-rho", "--latent-x", dest="latent_rho", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux-hidden", type=int, default=64)
    ap.add_argument("--critic-hidden", type=int, default=128)
    ap.add_argument("--count-pseudocount", type=float, default=0.5)
    ap.add_argument("--relative-pseudocount", type=float, default=1e-6)
    ap.add_argument("--init-coord-scale", type=float, default=0.5)
    ap.add_argument("--init-concentration", type=float, default=50.0)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--beta-max", type=float, default=1.0)
    ap.add_argument("--kl-warmup-frac", type=float, default=0.25)
    ap.add_argument("--free-bits", type=float, default=0.02)
    ap.add_argument("--alpha-d", type=float, default=1.0)
    ap.add_argument("--alpha-y", type=float, default=10.0)
    ap.add_argument("--unlabelled-y-prior-weight", type=float, default=1.0)
    ap.add_argument("--concentration-l2", type=float, default=1e-4)

    # TAXI extras
    ap.add_argument(
        "--lambda-cond-critic", type=float, default=0.1,
        help="Weight on the conditional study critic on z_rho.",
    )
    ap.add_argument(
        "--lambda-cond-coral", type=float, default=0.1,
        help="Weight on class-conditional CORAL on z_rho.",
    )
    ap.add_argument(
        "--lambda-philr-smooth", type=float, default=0.0,
        help="Weight on the shift-invariant PhILR-coord smoothness penalty.",
    )
    ap.add_argument(
        "--lambda-orth", type=float, default=0.0,
        help="Weight on the conditional cross-covariance penalty between z_tau and z_rho.",
    )
    ap.add_argument(
        "--lambda-tau-aux", type=float, default=0.0,
        help="Weight on the auxiliary CE on z_tau alone (protects the taxonomy/class channel).",
    )
    ap.add_argument("--grl-lambda", type=float, default=1.0)
    ap.add_argument(
        "--no-condition-orth-on-domain", action="store_true",
        help="If set, the orthogonality penalty is class-conditional but domain-pooled.",
    )

    ap.add_argument("--keep-prefixes", action="store_true")
    ap.add_argument("--taxonomy-has-header", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_optuna_cli_args(ap)
    return ap


def _epoch_pass(
    model, loader, optimizer, *, train: bool, beta: float, args,
    device: torch.device, likelihood: str, validate_counts: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    totals: Dict[str, float] = {}
    n_total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_leaf, domain, klass in loader:
            x_leaf = x_leaf.to(device, non_blocking=True)
            domain = domain.to(device, non_blocking=True)
            klass = klass.to(device, non_blocking=True)
            loss, metrics = model.loss(
                x_leaf, domain, klass=klass,
                likelihood=likelihood, data_kind=args.data_kind,
                beta=beta,
                alpha_d=args.alpha_d, alpha_y=args.alpha_y,
                unlabelled_y_prior_weight=args.unlabelled_y_prior_weight,
                free_bits=args.free_bits,
                concentration_l2=args.concentration_l2,
                validate_counts=validate_counts,
                lambda_cond_critic=args.lambda_cond_critic,
                lambda_cond_coral=args.lambda_cond_coral,
                lambda_philr_smooth=args.lambda_philr_smooth,
                lambda_orth=args.lambda_orth,
                lambda_tau_aux=args.lambda_tau_aux,
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            bsz = x_leaf.size(0)
            n_total += bsz
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bsz
    return {k: v / max(1, n_total) for k, v in totals.items()}


def _train(
    args: argparse.Namespace,
    outdir: Path,
    *,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run one TAXI-Hyperbolic-PhILR-VAE training pass."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds, X_leaf, leaf_names, taxg = _build_dataset(args)
    device = torch.device(args.device)
    likelihood = args.likelihood
    validate_counts = likelihood not in {"philr_gaussian", "dirichlet_tree"}

    n_samples = X_leaf.size(0)
    train_idx, val_idx = split_train_val(n_samples, args.val_split, args.seed)
    train_ds = torch.utils.data.TensorDataset(
        X_leaf[train_idx], ds.domain[train_idx], ds.klass[train_idx],
    )
    val_ds = torch.utils.data.TensorDataset(
        X_leaf[val_idx], ds.domain[val_idx], ds.klass[val_idx],
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TAXIHyperbolicPhILRVAE(
        n_domains=len(ds.domain_classes),
        n_classes=len(ds.class_classes),
        taxg=taxg,
        curvature=args.curvature,
        hidden=tuple(args.hidden),
        latent_d=args.latent_d,
        latent_tau=args.latent_tau,
        latent_rho=args.latent_rho,
        dropout=args.dropout,
        aux_hidden=args.aux_hidden,
        critic_hidden=args.critic_hidden,
        count_pseudocount=args.count_pseudocount,
        relative_pseudocount=args.relative_pseudocount,
        default_likelihood=likelihood,
        init_coord_scale=args.init_coord_scale,
        init_concentration=args.init_concentration,
        grl_lambda=args.grl_lambda,
        condition_orth_on_domain=not args.no_condition_orth_on_domain,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )
    warmup = max(1, int(args.epochs * args.kl_warmup_frac))
    best_val = float("inf")
    no_improve = 0
    log_rows = []
    model_path = outdir / "model.pt"

    for ep in range(1, args.epochs + 1):
        beta = beta_schedule(ep, warmup, args.beta_max)
        tr = _epoch_pass(model, train_dl, optimizer, train=True, beta=beta,
                         args=args, device=device, likelihood=likelihood,
                         validate_counts=validate_counts)
        va = _epoch_pass(model, val_dl, optimizer, train=False, beta=beta,
                         args=args, device=device, likelihood=likelihood,
                         validate_counts=validate_counts)
        row = {"epoch": ep, "beta": float(beta)}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        row["train_recon"] = tr.get("reconstruction_nll", float("nan"))
        row["val_recon"] = va.get("reconstruction_nll", float("nan"))
        log_rows.append(row)
        val_nll = va.get("reconstruction_nll", float("inf"))
        if not np.isfinite(val_nll):
            print("Non-finite val loss; stopping.")
            break
        scheduler.step(val_nll)
        improved = val_nll + 1e-9 < best_val
        if improved:
            best_val = val_nll
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if args.early_stop > 0 and no_improve >= args.early_stop:
                print("Early stopping.")
                break
        if verbose:
            print(
                f"ep {ep:03d} | beta={beta:.3f} | val_recon={val_nll:.3f} "
                f"cond_critic={va.get('cond_critic', 0):.3f} "
                f"cond_coral={va.get('cond_coral', 0):.3f} "
                f"orth={va.get('orth', 0):.3f}"
            )

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    cfg = {
        "model_type": "taxi-hyp-philrvae",
        "likelihood": likelihood,
        "data_kind": args.data_kind,
        "hidden": list(args.hidden),
        "curvature": args.curvature,
        # Persist BOTH naming conventions so loso_strict_encode can rebuild
        # the model regardless of whether the caller asks for z_y/z_x or
        # z_tau/z_rho.
        "latent_d": args.latent_d,
        "latent_y": args.latent_tau,
        "latent_x": args.latent_rho,
        "latent_tau": args.latent_tau,
        "latent_rho": args.latent_rho,
        "dropout": args.dropout,
        "aux_hidden": args.aux_hidden,
        "critic_hidden": args.critic_hidden,
        "count_pseudocount": args.count_pseudocount,
        "relative_pseudocount": args.relative_pseudocount,
        "init_coord_scale": args.init_coord_scale,
        "init_concentration": args.init_concentration,
        "grl_lambda": args.grl_lambda,
        "lambda_cond_critic": args.lambda_cond_critic,
        "lambda_cond_coral": args.lambda_cond_coral,
        "lambda_philr_smooth": args.lambda_philr_smooth,
        "lambda_orth": args.lambda_orth,
        "lambda_tau_aux": args.lambda_tau_aux,
        "condition_orth_on_domain": not args.no_condition_orth_on_domain,
        "n_domains": len(ds.domain_classes),
        "n_classes": len(ds.class_classes),
        "domain_classes": ds.domain_classes,
        "class_classes": ds.class_classes,
        "feature_clades": leaf_names,
        "model_kwargs": {
            "keep_prefixes": bool(args.keep_prefixes),
            "taxonomy_has_header": bool(args.taxonomy_has_header),
        },
    }
    embeddings = encode_full_dataset(
        model=model,
        encode_fn=lambda x: model.encode(x, data_kind=args.data_kind),
        inputs=X_leaf, batch_size=128, device=device,
    )
    recon_parts = []
    leaf_totals = X_leaf.sum(dim=1, keepdim=True).clamp(min=1.0)
    with torch.no_grad():
        for start in range(0, X_leaf.size(0), 128):
            batch = X_leaf[start:start + 128].to(device)
            enc = model.encode(batch, data_kind=args.data_kind)
            lp = model.decode_parts(enc["mu_d"], enc["mu_y"], enc["mu_x"])["leaf_prob"]
            lib = leaf_totals[start:start + 128].to(device)
            recon_parts.append((lp * lib).cpu().numpy())
    recon = np.concatenate(recon_parts, axis=0) if recon_parts else None

    save_diva_outputs(
        outdir=outdir,
        sample_ids=ds.sample_ids,
        feature_clades=leaf_names,
        embeddings=embeddings,
        recon=recon,
        log_rows=log_rows,
        config=cfg,
    )
    if verbose:
        print(f"Best val_recon: {best_val:.4f}")
    return {"best_val": float(best_val), "log_rows": log_rows}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.optuna:
        run_diva_optuna(
            args,
            lambda a, outdir, *, verbose=True: _train(a, outdir, verbose=verbose),
            default_search_space=DEFAULT_TAXI_HYP_PHILRVAE_SEARCH_SPACE,
        )
        return
    _train(args, Path(args.outdir), verbose=True)


if __name__ == "__main__":
    main()
