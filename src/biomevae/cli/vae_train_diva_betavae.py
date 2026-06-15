"""Training CLI for DIVA β-VAE (non-taxonomy backbone).

Wraps :class:`biomevae.models.diva_betavae.DIVABetaVAE` behind the same
calling convention as the other ``biomevae-train-*`` CLIs.  Reconstruction
is MSE / MAE / Huber on log1p-counts (matching the plain ``biomevae-train``
β-VAE preprocessing); KL warmup / β-max defaults mirror the values in
``vae_train.py``.

Usage::

    biomevae-train-diva-beta-vae \\
        --input  merged_sgb_table.tsv \\
        --metadata merged_sample_metadata.tsv \\
        --label-col disease --outdir out/diva-beta-vae
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any, Dict

import torch
import torch.utils.data

from biomevae.cli._diva_common import (
    add_optuna_cli_args,
    build_diva_dataset,
    diva_train_loop,
    encode_full_dataset,
    run_diva_optuna,
    save_diva_outputs,
    split_train_val,
)
from biomevae.losses import reconstruction_loss
from biomevae.models.diva_betavae import DIVABetaVAE
from biomevae.utils import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-diva-beta-vae")
    ap.add_argument("--input", required=True, help="Merged sgb_table.tsv")
    ap.add_argument("--metadata", required=True, help="Merged sample_metadata.tsv")
    ap.add_argument("--label-col", default="disease")
    ap.add_argument("--study-col", default="study_name")
    ap.add_argument("--outdir", required=True)

    # Architecture
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    ap.add_argument("--latent-d", type=int, default=4)
    ap.add_argument("--latent-y", type=int, default=8)
    ap.add_argument("--latent-x", type=int, default=8)
    ap.add_argument(
        "--activation", default="leakyrelu",
        choices=["leakyrelu", "gelu", "relu"],
    )
    ap.add_argument("--layer-norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--aux-hidden", type=int, default=64)

    # Optimisation (defaults match biomevae-train / β-VAE convention)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--beta-max", type=float, default=0.05)
    ap.add_argument(
        "--kl-warmup-frac", type=float, default=0.5,
        help=(
            "KL warmup as a fraction of --epochs.  The β-VAE pipeline "
            "uses a deliberately slow schedule (default 0.5) to keep β "
            "small early and avoid posterior collapse."
        ),
    )
    ap.add_argument("--free-bits", type=float, default=0.02)
    ap.add_argument("--alpha-d", type=float, default=1.0)
    ap.add_argument("--alpha-y", type=float, default=10.0)
    ap.add_argument(
        "--recon-kind", default="mae",
        choices=["mse", "mae", "huber"],
        help="Reconstruction loss flavour (default: mae).",
    )
    ap.add_argument("--huber-delta", type=float, default=1.0)

    # Preprocessing
    ap.add_argument(
        "--log1p", dest="log1p", action="store_true", default=True,
        help="Apply log1p to inputs (default; kept for config compatibility).",
    )
    ap.add_argument(
        "--no-log1p", dest="log1p", action="store_false",
        help="Skip log1p of inputs (default: applied).",
    )

    # Misc
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    add_optuna_cli_args(ap)
    return ap


def _train_diva_betavae(
    args: argparse.Namespace,
    outdir: Path,
    *,
    verbose: bool = True,
    argv: list[str] | None = None,
) -> Dict[str, Any]:
    """Run one DIVA β-VAE training pass and write artefacts to ``outdir``.

    Factored out of ``main`` so the Optuna runner in ``_diva_common`` can
    drive multiple trials with mutated argparse namespaces.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_global_seed(int(args.seed))

    # ------- dataset -------------------------------------------------
    # ``build_diva_dataset`` returns raw counts in ``x_raw``; we apply
    # log1p ourselves so the same CLI works for already-transformed
    # inputs via ``--no-log1p``.
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
            f"DIVA training requires >=2 classes; found {n_classes} "
            f"({ds.class_classes!r})."
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
    model = DIVABetaVAE(
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
    ).to(device)

    # ------- training ------------------------------------------------
    def forward_fn(model_, batch, *, free_bits):
        x, dom, klass = batch
        return model_(x, dom, klass, free_bits=free_bits)

    def recon_nll(out):
        return reconstruction_loss(
            out["x"], out["recon"],
            kind=args.recon_kind, huber_delta=args.huber_delta,
            per_feature="sum",
        )

    res = diva_train_loop(
        model=model,
        forward_fn=forward_fn,
        recon_nll=recon_nll,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        beta_max=args.beta_max,
        kl_warmup_frac=args.kl_warmup_frac,
        free_bits=float(args.free_bits),
        alpha_d=float(args.alpha_d),
        alpha_y=float(args.alpha_y),
        grad_clip=args.grad_clip,
        early_stop=args.early_stop,
        outdir=outdir,
        device=device,
        diva_combine=DIVABetaVAE.diva_loss_combine,
        verbose=verbose,
    )

    # ------- embed + reconstruct -------------------------------------
    def encode_fn(batch_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = model.encode(batch_x)
        return {"mu_d": enc["mu_d"], "mu_y": enc["mu_y"], "mu_x": enc["mu_x"]}

    embeddings = encode_full_dataset(
        model=model, encode_fn=encode_fn,
        inputs=x_in, batch_size=128, device=device,
    )

    model.eval()
    with torch.no_grad():
        mu_d = torch.from_numpy(embeddings["mu_d"]).to(device)
        mu_y = torch.from_numpy(embeddings["mu_y"]).to(device)
        mu_x = torch.from_numpy(embeddings["mu_x"]).to(device)
        recon = model.reconstruct(mu_d, mu_y, mu_x).cpu().numpy()

    config: Dict[str, Any] = {
        "model_type": "diva-beta-vae",
        "hidden": list(args.hidden),
        "latent_d": args.latent_d,
        "latent_y": args.latent_y,
        "latent_x": args.latent_x,
        "latent_dim": args.latent_d + args.latent_y + args.latent_x,
        "activation": args.activation,
        "layer_norm": bool(args.layer_norm),
        "dropout": args.dropout,
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
        "best_val_nll": res.best_val,
        "argv": shlex.join(argv) if argv is not None else None,
    }
    save_diva_outputs(
        outdir=outdir,
        sample_ids=ds.sample_ids,
        feature_clades=ds.feature_clades,
        embeddings=embeddings,
        recon=recon,
        log_rows=res.log_rows,
        config=config,
    )
    if verbose:
        print(f"\nDIVA β-VAE best val recon: {res.best_val:.6f}")
    return {"best_val": float(res.best_val), "log_rows": res.log_rows}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.optuna:
        run_diva_optuna(
            args,
            lambda a, outdir, *, verbose=True: _train_diva_betavae(
                a, outdir, verbose=verbose, argv=argv,
            ),
        )
        return

    _train_diva_betavae(
        args, Path(args.outdir), verbose=True, argv=argv,
    )


if __name__ == "__main__":
    main()
