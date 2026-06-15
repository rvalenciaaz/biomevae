"""Training script for TreeDTM-VAE (Dirichlet-tree-multinomial / Dirichlet-tree).

Usage::

    biomevae-train-tree-dtm --input sgb_table.tsv --taxonomy phyla.tsv --outdir out/

Two-term loss: tree-likelihood NLL + beta * KL + small concentration L2.
Hierarchical consistency is structurally guaranteed by the grouped tree-softmax
decoder; sample-level dispersion is captured by per-clade Dirichlet
concentrations.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data

from biomevae.losses import beta_schedule
from biomevae.models.tree_dtm_vae import (
    TreeDTMVAE,
    build_tree_topology,
    build_treevae_dataset,
)
from biomevae.optuna_utils import filter_trial_params, load_search_space


LIKELIHOOD_CHOICES = ("dirichlet_tree_multinomial", "tree_multinomial", "dirichlet_tree")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-tree-dtm")
    ap.add_argument("--input", required=True, help="Path to feature x sample table (TSV)")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy table (TSV)")
    ap.add_argument("--outdir", required=True)

    ap.add_argument(
        "--data-kind", choices=("counts", "relative"), default="relative",
        help="counts -> integer-validated; relative -> closed compositions",
    )
    ap.add_argument(
        "--likelihood", choices=LIKELIHOOD_CHOICES, default="dirichlet_tree",
    )

    # Architecture
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--encoder-layers", type=int, default=2)
    ap.add_argument("--decoder-hidden", type=int, default=256)
    ap.add_argument("--decoder-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--encoder-pseudocount", type=float, default=0.5)
    ap.add_argument("--init-concentration", type=float, default=50.0)

    # Optimisation
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--beta-max", type=float, default=1.0)
    ap.add_argument("--kl-warmup-frac", type=float, default=0.25)
    ap.add_argument("--free-bits", type=float, default=0.02)
    ap.add_argument("--concentration-l2", type=float, default=1e-4)

    # Misc
    ap.add_argument("--keep-prefixes", action="store_true")
    ap.add_argument("--taxonomy-has-header", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--min-matched-fraction", type=float, default=0.95,
        help="Minimum fraction of tree leaves matched in the table.",
    )
    ap.add_argument(
        "--allow-missing-leaves", action="store_true",
        help="Fill unmatched tree leaves with zero instead of failing.",
    )

    # Optuna
    ap.add_argument("--optuna", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=30)
    ap.add_argument("--optuna-config", type=str, default=None)
    return ap


def _make_loaders(
    X_nodes: torch.Tensor,
    X_leaves: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
):
    train_ds = torch.utils.data.TensorDataset(X_nodes[train_idx], X_leaves[train_idx])
    val_ds = torch.utils.data.TensorDataset(X_nodes[val_idx], X_leaves[val_idx])
    tl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return tl, vl


def _train(
    model: TreeDTMVAE,
    loader,
    *,
    val_loader=None,
    epochs: int,
    lr: float,
    beta_max: float,
    kl_warmup_frac: float,
    grad_clip: float,
    likelihood: str,
    free_bits: float,
    concentration_l2: float,
    validate_counts: bool,
    early_stop: int,
    outdir: Path,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )
    warmup = max(1, int(epochs * kl_warmup_frac))
    log_rows: list[dict] = []
    best_val = float("inf")
    no_improve = 0
    model_path = outdir / "model.pt"

    for ep in range(1, epochs + 1):
        beta = beta_schedule(ep, warmup, beta_max)

        model.train()
        t_recon = t_kl = t_loss = 0.0
        n = 0
        for x_nodes, _ in loader:
            x_nodes = x_nodes.to(device, non_blocking=True)
            out = model(x_nodes)
            loss, metrics = model.loss(
                x_nodes,
                outputs=out,
                likelihood=likelihood,
                beta=beta,
                free_bits=free_bits,
                concentration_l2=concentration_l2,
                validate_counts=validate_counts,
            )
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            bsz = x_nodes.size(0)
            t_recon += float(metrics["reconstruction_nll"]) * bsz
            t_kl += float(metrics["kl"]) * bsz
            t_loss += float(loss) * bsz
            n += bsz

        row: dict = {
            "epoch": ep,
            "beta": float(beta),
            "train_loss": t_loss / n,
            "train_nll": t_recon / n,
            "train_kl": t_kl / n,
            "train_recon": t_recon / n,
        }

        if val_loader is not None:
            model.eval()
            v_recon = v_kl = v_loss = 0.0
            v_n = 0
            with torch.no_grad():
                for x_nodes, _ in val_loader:
                    x_nodes = x_nodes.to(device, non_blocking=True)
                    out = model(x_nodes)
                    loss, metrics = model.loss(
                        x_nodes,
                        outputs=out,
                        likelihood=likelihood,
                        beta=beta,
                        free_bits=free_bits,
                        concentration_l2=concentration_l2,
                        validate_counts=validate_counts,
                    )
                    bsz = x_nodes.size(0)
                    v_recon += float(metrics["reconstruction_nll"]) * bsz
                    v_kl += float(metrics["kl"]) * bsz
                    v_loss += float(loss) * bsz
                    v_n += bsz
            val_nll = v_recon / v_n
            row["val_loss"] = v_loss / v_n
            row["val_nll"] = val_nll
            row["val_kl"] = v_kl / v_n
            row["val_recon"] = val_nll

            if not np.isfinite(row["train_loss"]) or not np.isfinite(row["val_loss"]):
                if verbose:
                    print("Stopping: non-finite loss.")
                break
            scheduler.step(val_nll)

            improved = val_nll + 1e-9 < best_val
            if early_stop > 0:
                if improved:
                    best_val = val_nll
                    no_improve = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    no_improve += 1
                    if no_improve >= early_stop:
                        if verbose:
                            print("Early stopping.")
                        break
            else:
                if improved:
                    best_val = val_nll
                torch.save(model.state_dict(), model_path)

        log_rows.append(row)
        if verbose:
            msg = (
                f"ep {ep:03d} | β={beta:.3f} | loss={row['train_loss']:.2f} "
                f"nll={row['train_nll']:.2f} kl={row['train_kl']:.2f}"
            )
            if val_loader is not None:
                msg += f" | val={row['val_loss']:.2f}"
            print(msg)

        if val_loader is None and ep % 25 == 0:
            torch.save(model.state_dict(), outdir / f"model_epoch{ep:03d}.pt")

    if val_loader is not None and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return {"log_rows": log_rows, "best_val": best_val}


def _suggest_params(trial, base, frozen=None):
    frozen = frozen or {}
    p = copy.deepcopy(base)
    p["hidden"] = frozen.get(
        "hidden", trial.suggest_categorical("hidden", [128, 192, 256, 384, 512])
    )
    p["latent_dim"] = frozen.get(
        "latent_dim",
        trial.suggest_categorical("latent_dim", [8, 16, 24, 32, 48, 64]),
    )
    p["encoder_layers"] = frozen.get(
        "encoder_layers", trial.suggest_categorical("encoder_layers", [1, 2, 3])
    )
    p["decoder_hidden"] = frozen.get(
        "decoder_hidden",
        trial.suggest_categorical("decoder_hidden", [128, 256, 384]),
    )
    p["decoder_layers"] = frozen.get(
        "decoder_layers", trial.suggest_categorical("decoder_layers", [1, 2, 3])
    )
    p["dropout"] = frozen.get("dropout", trial.suggest_float("dropout", 0.0, 0.3))
    p["lr"] = frozen.get("lr", trial.suggest_float("lr", 1e-4, 5e-3, log=True))
    p["beta_max"] = frozen.get(
        "beta_max", trial.suggest_float("beta_max", 0.01, 2.0, log=True)
    )
    p["kl_warmup_frac"] = frozen.get(
        "kl_warmup_frac", trial.suggest_float("kl_warmup_frac", 0.1, 0.5)
    )
    p["free_bits"] = frozen.get(
        "free_bits", trial.suggest_float("free_bits", 0.0, 0.1)
    )
    p["init_concentration"] = frozen.get(
        "init_concentration", trial.suggest_float("init_concentration", 5.0, 200.0, log=True)
    )
    p["batch_size"] = frozen.get(
        "batch_size", trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    )
    return p


def _build_trial_params(trial, base, config=None):
    frozen: dict = {}
    if config:
        for k, spec in config.items():
            if isinstance(spec, dict) and "method" in spec:
                continue
            frozen[k] = spec
    params = _suggest_params(trial, base, frozen)
    if config:
        from biomevae.optuna_utils import _suggest_from_config, _assign_nested

        for key, value in _suggest_from_config(trial, config).items():
            _assign_nested(params, key, value)
    return params


def _save_outputs(
    model: TreeDTMVAE,
    X_nodes: torch.Tensor,
    X_leaves: torch.Tensor,
    sample_ids,
    leaf_names,
    params: Dict[str, Any],
    res: Dict[str, Any],
    outdir: Path,
    device: torch.device,
) -> None:
    import pandas as pd

    pd.DataFrame(res["log_rows"]).to_csv(
        outdir / "training_log.tsv", sep="\t", index=False
    )

    cfg = {
        "model_type": "tree-dtm-vae",
        "likelihood": params.get("likelihood"),
        "data_kind": params.get("data_kind"),
        "hidden": params["hidden"],
        "latent_dim": params["latent_dim"],
        "encoder_layers": params.get("encoder_layers", 2),
        "decoder_hidden": params["decoder_hidden"],
        "decoder_layers": params.get("decoder_layers", 2),
        "dropout": params["dropout"],
        "encoder_pseudocount": params.get("encoder_pseudocount", 0.5),
        "init_concentration": params.get("init_concentration", 50.0),
        "feature_clades": leaf_names,
        "model_kwargs": {
            "keep_prefixes": params.get("keep_prefixes", False),
            "taxonomy_has_header": params.get("taxonomy_has_header", False),
        },
    }
    with (outdir / "config.json").open("w") as fh:
        json.dump(cfg, fh, indent=2)

    ds = torch.utils.data.TensorDataset(X_nodes, X_leaves)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    model.eval()
    emb_parts, recon_parts = [], []
    with torch.no_grad():
        for x_nodes, _ in loader:
            x_nodes = x_nodes.to(device, non_blocking=True)
            mu_z, _ = model.encode(x_nodes)
            emb_parts.append(mu_z.cpu().numpy())
            leaf_prob = model.decode(mu_z)["leaf_prob"]
            recon_parts.append(leaf_prob.cpu().numpy())

    emb = np.concatenate(emb_parts)
    recon = np.concatenate(recon_parts)
    pd.DataFrame(
        emb, index=sample_ids,
        columns=[f"z{i}" for i in range(emb.shape[1])],
    ).to_csv(outdir / "embeddings.tsv", sep="\t")
    pd.DataFrame(recon, index=sample_ids, columns=leaf_names).to_csv(
        outdir / "recon.tsv", sep="\t"
    )


def _run_optuna(args, taxg, topo, X_nodes, X_leaves, sample_ids, leaf_names, base_params):
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna not installed. `pip install biomevae[optuna]`."
        ) from exc

    config = None
    if args.optuna_config:
        config = load_search_space(args.optuna_config)

    study_dir = os.path.join(args.outdir, "optuna_trials")
    os.makedirs(study_dir, exist_ok=True)

    n_samples = len(sample_ids)
    device = torch.device(args.device)
    likelihood = base_params["likelihood"]
    validate_counts = bool(base_params.get("validate_counts", likelihood != "dirichlet_tree"))

    def objective(trial):
        params = _build_trial_params(trial, base_params, config)
        trial_out = os.path.join(study_dir, f"trial_{trial.number:04d}")
        os.makedirs(trial_out, exist_ok=True)
        seed = args.seed + trial.number
        torch.manual_seed(seed)
        np.random.seed(seed)

        idx = np.random.permutation(n_samples)
        n_val = max(1, int(n_samples * params["val_split"]))
        tl, vl = _make_loaders(
            X_nodes, X_leaves, idx[n_val:], idx[:n_val], params["batch_size"],
        )

        model = TreeDTMVAE(
            topo,
            hidden=params["hidden"],
            latent_dim=params["latent_dim"],
            encoder_layers=params.get("encoder_layers", 2),
            decoder_hidden=params["decoder_hidden"],
            decoder_layers=params.get("decoder_layers", 2),
            dropout=params["dropout"],
            encoder_pseudocount=params.get("encoder_pseudocount", 0.5),
            init_concentration=params.get("init_concentration", 50.0),
            likelihood=likelihood,
        ).to(device)

        try:
            res = _train(
                model, tl, val_loader=vl,
                epochs=params["epochs"], lr=params["lr"],
                beta_max=params["beta_max"],
                kl_warmup_frac=params["kl_warmup_frac"],
                grad_clip=params.get("grad_clip", 5.0),
                likelihood=likelihood,
                free_bits=params.get("free_bits", 0.0),
                concentration_l2=params.get("concentration_l2", 1e-4),
                validate_counts=validate_counts,
                early_stop=params["early_stop"],
                outdir=Path(trial_out), device=device, verbose=False,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float("inf")

        trial.set_user_attr("params", filter_trial_params(params))
        trial.set_user_attr("full_params", copy.deepcopy(params))
        trial.set_user_attr("seed", seed)
        if not np.isfinite(res["best_val"]):
            return float("inf")
        return res["best_val"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.optuna_trials, catch=(Exception,))

    try:
        best = study.best_trial
    except ValueError as exc:
        raise SystemExit(
            f"Optuna: every trial failed before recording a val loss "
            f"({exc}). Inspect per-trial logs under {study_dir}."
        ) from exc
    if not np.isfinite(best.value):
        raise SystemExit("No finite val losses found.")
    bp = copy.deepcopy(best.user_attrs["full_params"])
    bs = best.user_attrs["seed"]

    retrain_epochs = bp["epochs"] * 2
    torch.manual_seed(bs)
    np.random.seed(bs)
    idx = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * bp["val_split"]))
    tl, vl = _make_loaders(X_nodes, X_leaves, idx[n_val:], idx[:n_val], bp["batch_size"])

    model = TreeDTMVAE(
        topo,
        hidden=bp["hidden"], latent_dim=bp["latent_dim"],
        encoder_layers=bp.get("encoder_layers", 2),
        decoder_hidden=bp["decoder_hidden"],
        decoder_layers=bp.get("decoder_layers", 2),
        dropout=bp["dropout"],
        encoder_pseudocount=bp.get("encoder_pseudocount", 0.5),
        init_concentration=bp.get("init_concentration", 50.0),
        likelihood=likelihood,
    ).to(device)
    outdir = Path(args.outdir)
    res = _train(
        model, tl, val_loader=vl,
        epochs=retrain_epochs, lr=bp["lr"],
        beta_max=bp["beta_max"], kl_warmup_frac=bp["kl_warmup_frac"],
        grad_clip=bp.get("grad_clip", 5.0),
        likelihood=likelihood,
        free_bits=bp.get("free_bits", 0.0),
        concentration_l2=bp.get("concentration_l2", 1e-4),
        validate_counts=validate_counts,
        early_stop=bp["early_stop"],
        outdir=outdir, device=device, verbose=True,
    )
    _save_outputs(model, X_nodes, X_leaves, sample_ids, leaf_names, bp, res, outdir, device)

    from biomevae.optuna_utils import filter_trial_params as _fp
    _out = os.path.join(args.outdir, "optuna_best_params.json")
    with open(_out, "w") as fh:
        json.dump(_fp(bp), fh, indent=2)
    try:
        study.trials_dataframe().to_csv(
            os.path.join(args.outdir, "optuna_trials.csv"), index=False
        )
    except Exception:
        pass
    print(f"\nOptuna best trial #{best.number} | val={best.value:.6f}")
    print(f"Best val loss: {res['best_val']:.6f}")


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    outdir = Path(args.outdir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    taxg, topo, X_nodes, X_leaves, sample_ids, leaf_names, report = build_treevae_dataset(
        args.input,
        args.taxonomy,
        data_kind=args.data_kind,
        keep_prefixes=args.keep_prefixes,
        strict_alignment=True,
        allow_missing_leaves=args.allow_missing_leaves,
        min_matched_fraction=args.min_matched_fraction,
        taxonomy_has_header=args.taxonomy_has_header,
    )
    print(f"[align] {report.summary()}")

    device = torch.device(args.device)
    likelihood = args.likelihood
    validate_counts = likelihood != "dirichlet_tree"

    base_params = {
        "device": args.device,
        "model_type": "tree-dtm-vae",
        "likelihood": likelihood,
        "data_kind": args.data_kind,
        "hidden": args.hidden,
        "latent_dim": args.latent_dim,
        "encoder_layers": args.encoder_layers,
        "decoder_hidden": args.decoder_hidden,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "encoder_pseudocount": args.encoder_pseudocount,
        "init_concentration": args.init_concentration,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "beta_max": args.beta_max,
        "kl_warmup_frac": args.kl_warmup_frac,
        "free_bits": float(args.free_bits),
        "concentration_l2": float(args.concentration_l2),
        "validate_counts": validate_counts,
        "keep_prefixes": bool(args.keep_prefixes),
        "taxonomy_has_header": bool(args.taxonomy_has_header),
        "val_split": args.val_split,
        "early_stop": args.early_stop,
    }

    if args.optuna:
        _run_optuna(
            args, taxg, topo, X_nodes, X_leaves, sample_ids, leaf_names, base_params,
        )
        return

    n_samples = len(sample_ids)
    idx = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * args.val_split))
    tl, vl = _make_loaders(X_nodes, X_leaves, idx[n_val:], idx[:n_val], args.batch_size)

    model = TreeDTMVAE(
        topo,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        encoder_layers=args.encoder_layers,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        encoder_pseudocount=args.encoder_pseudocount,
        init_concentration=args.init_concentration,
        likelihood=likelihood,
    ).to(device)

    res = _train(
        model, tl, val_loader=vl,
        epochs=args.epochs, lr=args.lr,
        beta_max=args.beta_max, kl_warmup_frac=args.kl_warmup_frac,
        grad_clip=args.grad_clip,
        likelihood=likelihood,
        free_bits=float(args.free_bits),
        concentration_l2=float(args.concentration_l2),
        validate_counts=validate_counts,
        early_stop=args.early_stop,
        outdir=outdir, device=device,
    )
    torch.save(model.state_dict(), outdir / "model.pt")
    _save_outputs(
        model, X_nodes, X_leaves, sample_ids, leaf_names, base_params, res,
        outdir, device,
    )
    print(f"\nBest val loss: {res['best_val']:.6f}")


if __name__ == "__main__":
    main()
