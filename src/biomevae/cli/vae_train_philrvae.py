"""Training script for the compositional :class:`PhILRVAE`.

Usage::

    biomevae-train-philrvae \\
        --input sgb_table.tsv --taxonomy phyla.tsv --outdir out/philrvae \\
        --data-kind counts --likelihood dirichlet_tree_multinomial

The CLI builds a tree-aligned dataset via
:func:`biomevae.models.philrvae.build_philrvae_dataset` and trains with one
of the five compositional likelihoods exposed by ``PhILRVAE``:

* ``philr_gaussian`` (logistic-normal on the simplex via orthonormal ILR)
* ``multinomial``
* ``dirichlet_multinomial``
* ``dirichlet_tree_multinomial``
* ``dirichlet_tree``
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from biomevae.cli._diva_common import add_optuna_cli_args, run_diva_optuna
from biomevae.losses import beta_schedule
from biomevae.models.philrvae import PhILRVAE, build_philrvae_dataset


LIKELIHOOD_CHOICES = (
    "philr_gaussian",
    "multinomial",
    "dirichlet_multinomial",
    "dirichlet_tree_multinomial",
    "dirichlet_tree",
)


# Optuna search space tuned for the non-DIVA PhILR-VAE backbones.  Keys
# must match argparse attribute names (underscores, not hyphens) so the
# generic ``run_diva_optuna`` helper can ``setattr`` them on ``args``
# between trials.  Latent / batch ranges mirror the values used by the
# Tree-DTM-VAE search in ``vae_train_tree_dtm_vae.py`` for consistency.
DEFAULT_PHILRVAE_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "latent_dim": {"method": "suggest_categorical", "choices": [16, 24, 32, 48, 64]},
    "lr": {"method": "suggest_float", "low": 1e-4, "high": 5e-3, "log": True},
    "dropout": {"method": "suggest_float", "low": 0.0, "high": 0.3},
    "beta_max": {"method": "suggest_float", "low": 0.01, "high": 2.0, "log": True},
    "kl_warmup_frac": {"method": "suggest_float", "low": 0.1, "high": 0.5},
    "free_bits": {"method": "suggest_float", "low": 0.0, "high": 0.1},
    "init_concentration": {
        "method": "suggest_float", "low": 5.0, "high": 200.0, "log": True,
    },
    "batch_size": {"method": "suggest_categorical", "choices": [16, 32, 64, 128]},
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-philrvae")
    ap.add_argument("--input", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--data-kind", choices=("counts", "relative"), default="relative")
    ap.add_argument("--likelihood", choices=LIKELIHOOD_CHOICES, default="philr_gaussian")

    ap.add_argument("--hidden", type=int, nargs="+", default=(256, 128))
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
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
    ap.add_argument("--concentration-l2", type=float, default=1e-4)

    ap.add_argument("--keep-prefixes", action="store_true")
    ap.add_argument("--taxonomy-has-header", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-missing-leaves", action="store_true")
    ap.add_argument("--min-matched-fraction", type=float, default=0.95)
    add_optuna_cli_args(ap)
    return ap


def _train(
    args: argparse.Namespace,
    outdir: Path,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one PhILR-VAE training pass and write artefacts to ``outdir``.

    Returns ``{"best_val": float, "log_rows": list}`` so the generic
    ``run_diva_optuna`` helper can score trials and pick a winner.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    taxg, X_leaf, X_nodes, sample_ids, leaf_names, report = build_philrvae_dataset(
        args.input, args.taxonomy,
        data_kind=args.data_kind,
        keep_prefixes=args.keep_prefixes,
        taxonomy_has_header=args.taxonomy_has_header,
        allow_missing_leaves=args.allow_missing_leaves,
        min_matched_fraction=args.min_matched_fraction,
    )
    print(f"[align] {report.summary()}")

    device = torch.device(args.device)
    model = PhILRVAE(
        taxg,
        latent_dim=args.latent_dim,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
        count_pseudocount=args.count_pseudocount,
        relative_pseudocount=args.relative_pseudocount,
        default_likelihood=args.likelihood,
        init_coord_scale=args.init_coord_scale,
        init_concentration=args.init_concentration,
    ).to(device)

    n = X_leaf.size(0)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * args.val_split))
    train_t = X_leaf[idx[n_val:]]
    val_t = X_leaf[idx[:n_val]]
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_t),
        batch_size=args.batch_size, shuffle=True,
    )
    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_t),
        batch_size=args.batch_size, shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )
    warmup = max(1, int(args.epochs * args.kl_warmup_frac))
    validate_counts = args.likelihood not in {"philr_gaussian", "dirichlet_tree"}

    best_val = float("inf")
    no_improve = 0
    log_rows = []
    model_path = outdir / "model.pt"

    for ep in range(1, args.epochs + 1):
        beta = beta_schedule(ep, warmup, args.beta_max)
        # train
        model.train()
        t_recon = t_kl = t_loss = 0.0
        n_t = 0
        for (xb,) in train_dl:
            xb = xb.to(device, non_blocking=True)
            loss, metrics = model.loss(
                xb, likelihood=args.likelihood, data_kind=args.data_kind,
                beta=beta, free_bits=args.free_bits,
                concentration_l2=args.concentration_l2,
                validate_counts=validate_counts,
            )
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            bsz = xb.size(0)
            t_recon += float(metrics["reconstruction_nll"]) * bsz
            t_kl += float(metrics["kl"]) * bsz
            t_loss += float(loss) * bsz
            n_t += bsz

        # val
        model.eval()
        v_recon = v_kl = v_loss = 0.0
        n_v = 0
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(device, non_blocking=True)
                loss, metrics = model.loss(
                    xb, likelihood=args.likelihood, data_kind=args.data_kind,
                    beta=beta, free_bits=args.free_bits,
                    concentration_l2=args.concentration_l2,
                    validate_counts=validate_counts,
                )
                bsz = xb.size(0)
                v_recon += float(metrics["reconstruction_nll"]) * bsz
                v_kl += float(metrics["kl"]) * bsz
                v_loss += float(loss) * bsz
                n_v += bsz

        row = {
            "epoch": ep, "beta": float(beta),
            "train_loss": t_loss / n_t,
            "train_nll": t_recon / n_t,
            "train_recon": t_recon / n_t,
            "train_kl": t_kl / n_t,
            "val_loss": v_loss / n_v,
            "val_nll": v_recon / n_v,
            "val_recon": v_recon / n_v,
            "val_kl": v_kl / n_v,
        }
        log_rows.append(row)
        val_nll = row["val_nll"]
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
                f"ep {ep:03d} | beta={beta:.3f} | val_recon={val_nll:.4f} kl={row['val_kl']:.4f}"
            )

    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    model.eval()

    pd.DataFrame(log_rows).to_csv(outdir / "training_log.tsv", sep="\t", index=False)
    cfg = {
        "model_type": "philrvae",
        "likelihood": args.likelihood,
        "data_kind": args.data_kind,
        "hidden": list(args.hidden),
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "count_pseudocount": args.count_pseudocount,
        "relative_pseudocount": args.relative_pseudocount,
        "init_coord_scale": args.init_coord_scale,
        "init_concentration": args.init_concentration,
        "feature_clades": leaf_names,
        "model_kwargs": {
            "keep_prefixes": bool(args.keep_prefixes),
            "taxonomy_has_header": bool(args.taxonomy_has_header),
        },
    }
    with (outdir / "config.json").open("w") as fh:
        json.dump(cfg, fh, indent=2)

    # embeddings + recon
    ds = torch.utils.data.TensorDataset(X_leaf)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    emb_parts, recon_parts = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            mu, _ = model.encode(xb, data_kind=args.data_kind)
            emb_parts.append(mu.cpu().numpy())
            dec = model.decode(mu)
            lib = xb.sum(dim=1, keepdim=True).clamp(min=1.0)
            recon_parts.append((dec["leaf_prob"] * lib).cpu().numpy())
    emb = np.concatenate(emb_parts, axis=0)
    recon = np.concatenate(recon_parts, axis=0)
    pd.DataFrame(emb, index=sample_ids, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(
        outdir / "embeddings.tsv", sep="\t",
    )
    pd.DataFrame(recon, index=sample_ids, columns=leaf_names).to_csv(
        outdir / "recon.tsv", sep="\t",
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
            default_search_space=DEFAULT_PHILRVAE_SEARCH_SPACE,
        )
        return
    _train(args, Path(args.outdir), verbose=True)


if __name__ == "__main__":
    main()
