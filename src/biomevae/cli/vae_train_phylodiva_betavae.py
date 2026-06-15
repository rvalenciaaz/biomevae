"""Training CLI for PhyloDIVA β-VAE (non-tax backbone).

Mirrors :mod:`biomevae.cli.vae_train_phylodiva_tree_dtm_vae` and
:mod:`biomevae.cli.vae_train_phylodiva_hyp_philrvae`: a standalone train
loop calling ``model.loss(...)`` once per batch, a built-in Optuna
search space, and a constant GRL coefficient (no DANN sigmoid ramp).

PhyloDIVA extras applied on the plain β-VAE backbone:
* a gradient-reversed study critic on ``z_y``
* CORAL on ``z_x``

Brownian-motion smoothness is silently a no-op (no tree-structured
decoder), so this backbone serves as the non-tree reference upper bound.
No taxonomy is required — the β-VAE is tax-agnostic and the GRL critic
operates on the latent ``z_y`` rather than tree-aggregated features.
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data

from biomevae.cli._diva_common import (
    DEFAULT_DIVA_SEARCH_SPACE,
    add_optuna_cli_args,
    build_diva_dataset,
    encode_full_dataset,
    run_diva_optuna,
    save_diva_outputs,
    split_train_val,
)
from biomevae.losses import beta_schedule
from biomevae.models.phylodiva_betavae import PhyloDIVABetaVAE
from biomevae.utils import set_global_seed


# Default Optuna search space for the PhyloDIVA β-VAE backbone.  Built
# on the shared DIVA defaults and adds the two PhyloDIVA-specific
# regularisation knobs.  No ``lambda_bm`` (β-VAE has no tree decoder)
# and no ``lambda_gr_max`` / ``critic_hidden`` — the GRL coefficient is
# held constant and the critic head width is a CLI arg with a sensible
# default.  Mirrors the recipe used by phylodiva-tree-dtm /
# phylodiva-hyp-philrvae.
DEFAULT_PHYLODIVA_BETA_VAE_SEARCH_SPACE: Dict[str, Any] = {
    **DEFAULT_DIVA_SEARCH_SPACE,
    "lambda_critic": {
        "method": "suggest_float", "low": 1e-3, "high": 1.0, "log": True,
    },
    "lambda_coral": {
        "method": "suggest_float", "low": 1e-3, "high": 1.0, "log": True,
    },
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-phylodiva-beta-vae")
    ap.add_argument("--input", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--label-col", default="disease")
    ap.add_argument("--study-col", default="study_name")
    ap.add_argument("--outdir", required=True)

    # Architecture
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    ap.add_argument("--latent-d", type=int, default=2)
    ap.add_argument("--latent-y", type=int, default=8)
    ap.add_argument("--latent-x", type=int, default=8)
    ap.add_argument(
        "--activation", default="leakyrelu",
        choices=["leakyrelu", "gelu", "relu"],
    )
    ap.add_argument("--layer-norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--aux-hidden", type=int, default=64)
    ap.add_argument("--critic-hidden", type=int, default=64)

    # Optimisation
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--beta-max", type=float, default=0.05)
    ap.add_argument("--kl-warmup-frac", type=float, default=0.25)
    ap.add_argument("--free-bits", type=float, default=0.02)
    ap.add_argument("--alpha-d", type=float, default=0.0)
    ap.add_argument("--alpha-y", type=float, default=10.0)
    ap.add_argument(
        "--recon-kind", default="mae",
        choices=["mse", "mae", "huber"],
    )
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument(
        "--log1p", dest="log1p", action="store_true", default=True,
        help="Apply log1p to inputs (default; kept for config compatibility).",
    )
    ap.add_argument(
        "--no-log1p", dest="log1p", action="store_false",
        help="Skip log1p of inputs (default: applied).",
    )

    # PhyloDIVA extras (constant-λ, no DANN ramp).
    ap.add_argument(
        "--lambda-critic", type=float, default=0.1,
        help="Weight on the latent-space study critic CE on z_y.",
    )
    ap.add_argument(
        "--lambda-coral", type=float, default=0.1,
        help="Weight on the per-study CORAL penalty on z_x.",
    )
    ap.add_argument(
        "--grl-lambda", type=float, default=1.0,
        help="Constant gradient-reversal coefficient on the study critic.",
    )

    # Misc
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    add_optuna_cli_args(ap)
    return ap


def _epoch_pass(
    model: PhyloDIVABetaVAE,
    loader,
    optimizer,
    *,
    train: bool,
    beta: float,
    args,
    device: torch.device,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    totals: Dict[str, float] = {}
    n_total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, domain, klass in loader:
            x = x.to(device, non_blocking=True)
            domain = domain.to(device, non_blocking=True)
            klass = klass.to(device, non_blocking=True)
            loss, metrics = model.loss(
                x, domain, klass=klass,
                recon_kind=args.recon_kind,
                huber_delta=args.huber_delta,
                beta=beta,
                alpha_d=args.alpha_d,
                alpha_y=args.alpha_y,
                free_bits=args.free_bits,
                lambda_critic=args.lambda_critic,
                lambda_coral=args.lambda_coral,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            bsz = x.size(0)
            n_total += bsz
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bsz
    return {k: v / max(1, n_total) for k, v in totals.items()}


def _train(
    args: argparse.Namespace,
    outdir: Path,
    *,
    verbose: bool = True,
    argv: list[str] | None = None,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(args.seed))

    ds = build_diva_dataset(
        args.input, args.metadata,
        label_col=args.label_col, study_col=args.study_col,
        eps=0.0,
    )
    n_features = ds.x_raw.size(1)
    n_domains = len(ds.domain_classes)
    n_classes = len(ds.class_classes)
    if n_classes < 2:
        raise SystemExit(
            f"PhyloDIVA training requires >=2 classes; found {n_classes}."
        )

    x_in = torch.log1p(ds.x_raw) if args.log1p else ds.x_raw.clone()

    train_idx, val_idx = split_train_val(
        x_in.size(0), args.val_split, args.seed,
    )
    train_ds = torch.utils.data.TensorDataset(
        x_in[train_idx], ds.domain[train_idx], ds.klass[train_idx],
    )
    val_ds = torch.utils.data.TensorDataset(
        x_in[val_idx], ds.domain[val_idx], ds.klass[val_idx],
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True,
    )

    device = torch.device(args.device)
    model = PhyloDIVABetaVAE(
        input_dim=n_features,
        n_domains=n_domains,
        n_classes=n_classes,
        hidden=list(args.hidden),
        latent_d=args.latent_d,
        latent_y=args.latent_y,
        latent_x=args.latent_x,
        dropout=args.dropout,
        activation=args.activation,
        layer_norm=args.layer_norm,
        aux_hidden=args.aux_hidden,
        critic_hidden=args.critic_hidden,
    ).to(device)
    # Constant GRL coefficient — matches phylodiva-tree-dtm /
    # phylodiva-hyp-philrvae.  No DANN sigmoid ramp.
    model.critic.set_lambda(float(args.grl_lambda))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    warmup = max(1, int(args.epochs * args.kl_warmup_frac))
    best_val = float("inf")
    no_improve = 0
    log_rows: list[dict] = []
    model_path = outdir / "model.pt"

    for ep in range(1, args.epochs + 1):
        beta = beta_schedule(ep, warmup, args.beta_max)
        tr = _epoch_pass(
            model, train_loader, optimizer, train=True,
            beta=beta, args=args, device=device,
        )
        va = _epoch_pass(
            model, val_loader, optimizer, train=False,
            beta=beta, args=args, device=device,
        )
        row = {"epoch": ep, "beta": float(beta)}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        row["train_recon"] = tr.get("reconstruction_nll", float("nan"))
        row["val_recon"] = va.get("reconstruction_nll", float("nan"))
        log_rows.append(row)

        val_nll = va.get("reconstruction_nll", float("inf"))
        if not np.isfinite(val_nll):
            if verbose:
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
                if verbose:
                    print("Early stopping.")
                break

        if verbose:
            print(
                f"ep {ep:03d} | beta={beta:.3f} | val_recon={val_nll:.3f} "
                f"critic={va.get('critic', 0):.3f} coral={va.get('coral', 0):.3f}"
            )

    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    model.eval()

    def encode_fn(batch_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = model.encode(batch_x)
        return {"mu_d": enc["mu_d"], "mu_y": enc["mu_y"], "mu_x": enc["mu_x"]}

    embeddings = encode_full_dataset(
        model=model, encode_fn=encode_fn,
        inputs=x_in, batch_size=128, device=device,
    )

    with torch.no_grad():
        mu_d = torch.from_numpy(embeddings["mu_d"]).to(device)
        mu_y = torch.from_numpy(embeddings["mu_y"]).to(device)
        mu_x = torch.from_numpy(embeddings["mu_x"]).to(device)
        recon = model.reconstruct(mu_d, mu_y, mu_x).cpu().numpy()

    config: Dict[str, Any] = {
        "model_type": "phylodiva-beta-vae",
        "hidden": list(args.hidden),
        "latent_d": args.latent_d,
        "latent_y": args.latent_y,
        "latent_x": args.latent_x,
        "latent_dim": args.latent_d + args.latent_y + args.latent_x,
        "activation": args.activation,
        "layer_norm": bool(args.layer_norm),
        "dropout": args.dropout,
        "aux_hidden": args.aux_hidden,
        "critic_hidden": args.critic_hidden,
        "log1p": bool(args.log1p),
        "recon_kind": args.recon_kind,
        "huber_delta": args.huber_delta,
        "n_domains": n_domains,
        "n_classes": n_classes,
        "domain_classes": ds.domain_classes,
        "class_classes": ds.class_classes,
        "feature_clades": ds.feature_clades,
        "label_col": args.label_col,
        "study_col": args.study_col,
        "alpha_d": args.alpha_d,
        "alpha_y": args.alpha_y,
        "free_bits": args.free_bits,
        "epochs": args.epochs,
        "lambda_critic": float(args.lambda_critic),
        "lambda_coral": float(args.lambda_coral),
        "grl_lambda": float(args.grl_lambda),
        "best_val_nll": best_val,
        "argv": shlex.join(argv) if argv is not None else None,
    }
    save_diva_outputs(
        outdir=outdir,
        sample_ids=ds.sample_ids,
        feature_clades=ds.feature_clades,
        embeddings=embeddings,
        recon=recon,
        log_rows=log_rows,
        config=config,
    )
    if verbose:
        print(f"\nPhyloDIVA β-VAE best val recon: {best_val:.6f}")
    return {"best_val": float(best_val), "log_rows": log_rows}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.optuna:
        run_diva_optuna(
            args,
            lambda a, outdir, *, verbose=True: _train(
                a, outdir, verbose=verbose, argv=argv,
            ),
            default_search_space=DEFAULT_PHYLODIVA_BETA_VAE_SEARCH_SPACE,
        )
        return
    _train(args, Path(args.outdir), verbose=True, argv=argv)


if __name__ == "__main__":
    main()
