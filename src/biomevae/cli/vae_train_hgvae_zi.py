from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from biomevae.models.hgvae_zi import (
    HGVAE_ZI,
    build_hgvae_zi_dataset,
    build_hgvae_zi_loader,
    hierarchical_consistency_loss,
    latent_smoothness_loss_from_affinity,
    load_sample_affinity_npy,
    zi_lognormal_nll,
)
from biomevae.optuna_utils import filter_trial_params, load_search_space


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-hgvae-zi")
    ap.add_argument("--input", required=True, help="Path to sgb_table.tsv")
    ap.add_argument("--taxonomy", required=True, help="Path to phyla.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-max", type=float, default=1.0)
    ap.add_argument("--beta-warmup-frac", type=float, default=0.3)
    ap.add_argument("--lambda-cons", type=float, default=1.0)
    ap.add_argument("--lambda-sample", type=float, default=0.0)
    ap.add_argument("--sample-affinity-npy", type=str, default=None)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--keep-prefixes", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

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


def _train(
    model: HGVAE_ZI,
    loader,
    *,
    val_loader=None,
    epochs: int,
    lr: float,
    beta_max: float,
    beta_warmup_frac: float,
    lambda_cons: float,
    lambda_sample: float,
    sample_affinity: Optional[np.ndarray],
    taxg,
    eps: float,
    early_stop: int = 0,
    outdir: Path,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_epochs = max(1, int(epochs * beta_warmup_frac))
    log_rows: list[dict[str, float]] = []
    best_val = float("inf")
    no_improve = 0
    model_path = outdir / "model.pt"

    for ep in range(1, epochs + 1):
        beta = beta_max * min(1.0, ep / warmup_epochs)

        # --- training ---
        model.train()
        tot = tot_recon = tot_kl = tot_cons = tot_smooth = 0.0
        n_batches = 0

        for data in loader:
            data = data.to(device)
            out = model(data)
            recon = zi_lognormal_nll(data.y, out["mu_log"], out["log_sig_log"], out["logit_pi"], eps=eps)
            kl = model.kl_standard_normal(out["mu"], out["logvar"])

            sig = torch.exp(out["log_sig_log"])
            mean_pos = torch.exp(out["mu_log"] + 0.5 * sig * sig).clamp_min(0.0)
            pi = torch.sigmoid(out["logit_pi"])
            x_pred = (1.0 - pi) * mean_pos
            cons = hierarchical_consistency_loss(
                x_pred=x_pred,
                batch=data.batch,
                children_of=taxg.children_of,
                internal_ids=taxg.internal_ids,
                eps=eps,
            )

            smooth = torch.tensor(0.0, device=device)
            if lambda_sample > 0 and sample_affinity is not None:
                idxs = data.sample_idx.view(-1).tolist()
                A_mb = torch.from_numpy(sample_affinity[np.ix_(idxs, idxs)]).to(device)
                smooth = latent_smoothness_loss_from_affinity(out["mu"], A_mb)

            loss = recon + beta * kl + lambda_cons * cons + lambda_sample * smooth
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot += float(loss.item())
            tot_recon += float(recon.item())
            tot_kl += float(kl.item())
            tot_cons += float(cons.item())
            tot_smooth += float(smooth.item())
            n_batches += 1

        row: dict[str, float] = {
            "epoch": float(ep),
            "beta": float(beta),
            "train_loss": tot / n_batches,
            "train_recon": tot_recon / n_batches,
            "train_kld": tot_kl / n_batches,
            "train_cons": tot_cons / n_batches,
            "train_smooth": tot_smooth / n_batches,
        }

        # --- validation ---
        if val_loader is not None:
            model.eval()
            v_tot = v_recon = v_kl = v_cons = 0.0
            v_batches = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data)
                    recon = zi_lognormal_nll(data.y, out["mu_log"], out["log_sig_log"], out["logit_pi"], eps=eps)
                    kl = model.kl_standard_normal(out["mu"], out["logvar"])

                    sig = torch.exp(out["log_sig_log"])
                    mean_pos = torch.exp(out["mu_log"] + 0.5 * sig * sig).clamp_min(0.0)
                    pi = torch.sigmoid(out["logit_pi"])
                    x_pred = (1.0 - pi) * mean_pos
                    cons = hierarchical_consistency_loss(
                        x_pred=x_pred,
                        batch=data.batch,
                        children_of=taxg.children_of,
                        internal_ids=taxg.internal_ids,
                        eps=eps,
                    )

                    v_loss = recon + beta * kl + lambda_cons * cons
                    v_tot += float(v_loss.item())
                    v_recon += float(recon.item())
                    v_kl += float(kl.item())
                    v_cons += float(cons.item())
                    v_batches += 1

            val_recon = v_recon / v_batches
            row["val_loss"] = v_tot / v_batches
            row["val_recon"] = val_recon
            row["val_kld"] = v_kl / v_batches
            row["val_cons"] = v_cons / v_batches

            if not np.isfinite(row["train_loss"]) or not np.isfinite(row["val_loss"]):
                if verbose:
                    print("Stopping early: non-finite train/val loss encountered.")
                break

            # Early stopping drives off val_recon — the non-stationary
            # β-weighted ELBO would otherwise make best-epoch selection
            # depend on where we are on the β warmup ramp.
            improved = val_recon + 1e-9 < best_val
            if early_stop > 0:
                if improved:
                    best_val = val_recon
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
                    best_val = val_recon
                torch.save(model.state_dict(), model_path)

        log_rows.append(row)
        if verbose:
            msg = (
                f"epoch {ep:03d} | beta={beta:.3g} | loss={row['train_loss']:.4f} recon={row['train_recon']:.4f} "
                f"kl={row['train_kld']:.4f} cons={row['train_cons']:.4f} smooth={row['train_smooth']:.4f}"
            )
            if val_loader is not None:
                msg += f" | val={row['val_loss']:.4f}"
            print(msg)

        if val_loader is None and ep % 25 == 0:
            torch.save(model.state_dict(), outdir / f"model_epoch{ep:03d}.pt")

    # Restore best model when using validation
    if val_loader is not None and model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    return {"log_rows": log_rows, "best_val": best_val}


def _suggest_hgvae_zi_params(
    trial,
    base: Dict[str, Any],
    frozen: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Suggest hyperparameters specific to HGVAE-ZI."""
    frozen = frozen or {}
    params = copy.deepcopy(base)

    params["hidden"] = frozen.get(
        "hidden",
        trial.suggest_categorical("hidden", [64, 96, 128, 192, 256]),
    )
    params["latent_dim"] = frozen.get(
        "latent_dim",
        trial.suggest_categorical("latent_dim", [2, 3, 4, 5, 6, 8, 10, 12, 16]),
    )
    params["lr"] = frozen.get(
        "lr",
        trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    )
    params["batch_size"] = frozen.get(
        "batch_size",
        trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    )
    params["beta_max"] = frozen.get(
        "beta_max",
        trial.suggest_float("beta_max", 0.01, 2.0, log=True),
    )
    params["beta_warmup_frac"] = frozen.get(
        "beta_warmup_frac",
        trial.suggest_float("beta_warmup_frac", 0.1, 0.5),
    )
    params["lambda_cons"] = frozen.get(
        "lambda_cons",
        trial.suggest_float("lambda_cons", 0.01, 10.0, log=True),
    )
    params["lambda_sample"] = frozen.get(
        "lambda_sample",
        trial.suggest_float("lambda_sample", 0.0, 1.0),
    )
    params["eps"] = frozen.get(
        "eps",
        trial.suggest_categorical("eps", [1e-8, 1e-7, 1e-6, 1e-5]),
    )
    return params


def _build_trial_params(
    trial,
    base: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frozen: Dict[str, Any] = {}
    if config:
        for key, spec in config.items():
            if isinstance(spec, dict) and "method" in spec:
                continue
            frozen[key] = spec
    params = _suggest_hgvae_zi_params(trial, base, frozen=frozen)
    if config:
        from biomevae.optuna_utils import _suggest_from_config, _assign_nested

        suggested = _suggest_from_config(trial, config)
        for key, value in suggested.items():
            _assign_nested(params, key, value)
    return params


def _prepare_base_params(args) -> Dict[str, Any]:
    return {
        "device": args.device,
        "model_type": "hgvae_zi",
        "hidden": args.hidden,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta_max": args.beta_max,
        "beta_warmup_frac": args.beta_warmup_frac,
        "lambda_cons": args.lambda_cons,
        "lambda_sample": args.lambda_sample,
        "eps": args.eps,
        "keep_prefixes": bool(args.keep_prefixes),
        "val_split": args.val_split,
        "early_stop": args.early_stop,
    }


def _save_optuna_artifacts(outdir: str, params: Dict[str, Any], study) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "optuna_best_params.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    try:
        df = study.trials_dataframe()
    except Exception:
        return
    df.to_csv(os.path.join(outdir, "optuna_trials.csv"), index=False)


def _run_optuna(
    args,
    taxg,
    dataset,
    sample_ids: List[str],
    sample_aff: Optional[np.ndarray],
    base_params: Dict[str, Any],
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

    rank_vocab = int(taxg.node_rank.max().item()) + 1
    n_samples = len(dataset)
    device = torch.device(args.device)

    def objective(trial):
        params = _build_trial_params(trial, base_params, config)
        trial_outdir = os.path.join(study_dir, f"trial_{trial.number:04d}")
        os.makedirs(trial_outdir, exist_ok=True)
        seed = args.seed + trial.number

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train/val split
        indices = np.random.permutation(n_samples)
        n_val = max(1, int(n_samples * params["val_split"]))
        val_idx = indices[:n_val].tolist()
        train_idx = indices[n_val:].tolist()

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = build_hgvae_zi_loader(train_subset, batch_size=params["batch_size"], shuffle=True)
        val_loader = build_hgvae_zi_loader(val_subset, batch_size=params["batch_size"], shuffle=False)

        model = HGVAE_ZI(hidden=params["hidden"], latent_dim=params["latent_dim"], rank_vocab=rank_vocab).to(device)

        try:
            res = _train(
                model,
                train_loader,
                val_loader=val_loader,
                epochs=params["epochs"],
                lr=params["lr"],
                beta_max=params["beta_max"],
                beta_warmup_frac=params["beta_warmup_frac"],
                lambda_cons=params["lambda_cons"],
                lambda_sample=params["lambda_sample"],
                sample_affinity=sample_aff,
                taxg=taxg,
                eps=params["eps"],
                early_stop=params["early_stop"],
                outdir=Path(trial_outdir),
                device=device,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trial.set_user_attr("oom", True)
            return float("inf")

        record = filter_trial_params(params)
        trial.set_user_attr("params", record)
        trial.set_user_attr("full_params", copy.deepcopy(params))
        trial.set_user_attr("seed", seed)

        if not np.isfinite(res["best_val"]):
            trial.set_user_attr("nonfinite", True)
            return float("inf")
        return res["best_val"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.optuna_trials)

    best_trial = study.best_trial
    if not np.isfinite(best_trial.value):
        raise SystemExit("Optuna did not find any finite validation losses; check input data and search space.")

    best_params = copy.deepcopy(best_trial.user_attrs["full_params"])
    best_record = copy.deepcopy(best_trial.user_attrs["params"])
    best_seed = best_trial.user_attrs["seed"]

    # Retrain best model with same seed (reproduces same train/val split)
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)

    indices = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * best_params["val_split"]))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = build_hgvae_zi_loader(train_subset, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = build_hgvae_zi_loader(val_subset, batch_size=best_params["batch_size"], shuffle=False)

    model = HGVAE_ZI(hidden=best_params["hidden"], latent_dim=best_params["latent_dim"], rank_vocab=rank_vocab).to(device)
    outdir = Path(args.outdir)
    res = _train(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=best_params["epochs"],
        lr=best_params["lr"],
        beta_max=best_params["beta_max"],
        beta_warmup_frac=best_params["beta_warmup_frac"],
        lambda_cons=best_params["lambda_cons"],
        lambda_sample=best_params["lambda_sample"],
        sample_affinity=sample_aff,
        taxg=taxg,
        eps=best_params["eps"],
        early_stop=best_params["early_stop"],
        outdir=outdir,
        device=device,
        verbose=True,
    )

    # Save artifacts
    import pandas as pd

    pd.DataFrame(res["log_rows"]).to_csv(outdir / "training_log.tsv", sep="\t", index=False)
    np.savetxt(outdir / "node_names.txt", np.asarray(taxg.node_names, dtype=object), fmt="%s")

    cfg = {
        "model_type": "hgvae_zi",
        "latent_dim": best_params["latent_dim"],
        "hidden": [best_params["hidden"]],
        "dropout": 0.0,
        "activation": "relu",
        "layer_norm": False,
        "input": args.input,
        "taxonomy": args.taxonomy,
        "feature_clades": dataset.sgb_ids,
        "model_kwargs": {
            "rank_vocab": rank_vocab,
            "eps": best_params["eps"],
            "keep_prefixes": bool(args.keep_prefixes),
        },
        "log1p": False,
        "standardize": False,
    }
    with (outdir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    embed_loader = build_hgvae_zi_loader(dataset, batch_size=best_params["batch_size"], shuffle=False)
    model.eval()
    emb_parts = []
    recon_parts = []
    with torch.no_grad():
        for data in embed_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            emb_parts.append(mu.cpu().numpy())
            xp = model.expected_abundance(data, mu)
            bsz = int(data.batch.max().item()) + 1
            n_nodes = xp.shape[0] // bsz
            leaf_ids = torch.tensor(taxg.leaf_ids, device=xp.device)
            leaf_vals = xp.view(bsz, n_nodes, 1)[:, leaf_ids, 0]
            recon_parts.append(leaf_vals.cpu().numpy())

    emb = np.concatenate(emb_parts, axis=0)
    recon = np.concatenate(recon_parts, axis=0)
    pd.DataFrame(emb, index=sample_ids, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(outdir / "embeddings.tsv", sep="\t")
    pd.DataFrame(recon, index=sample_ids, columns=dataset.sgb_ids).to_csv(outdir / "recon.tsv", sep="\t")

    _save_optuna_artifacts(args.outdir, best_record, study)
    print(f"\nOptuna best trial #{best_trial.number} | val={best_trial.value:.6f}")
    print(f"\nBest val loss: {res['best_val']:.6f}")


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    outdir = Path(args.outdir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    taxg, dataset, sample_ids = build_hgvae_zi_dataset(
        Path(args.input),
        Path(args.taxonomy),
        eps=args.eps,
        keep_prefixes=args.keep_prefixes,
    )
    rank_vocab = int(taxg.node_rank.max().item()) + 1

    sample_aff = None
    if args.sample_affinity_npy and args.lambda_sample > 0:
        sample_aff = load_sample_affinity_npy(Path(args.sample_affinity_npy))
        if sample_aff.shape[0] != len(sample_ids):
            raise SystemExit("sample affinity size must match number of samples in sgb_table.")

    if args.optuna:
        base_params = _prepare_base_params(args)
        _run_optuna(args, taxg, dataset, sample_ids, sample_aff, base_params)
        return

    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * args.val_split))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = build_hgvae_zi_loader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_hgvae_zi_loader(val_subset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = HGVAE_ZI(hidden=args.hidden, latent_dim=args.latent_dim, rank_vocab=rank_vocab).to(device)

    res = _train(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        beta_max=args.beta_max,
        beta_warmup_frac=args.beta_warmup_frac,
        lambda_cons=args.lambda_cons,
        lambda_sample=args.lambda_sample,
        sample_affinity=sample_aff,
        taxg=taxg,
        eps=args.eps,
        early_stop=args.early_stop,
        outdir=outdir,
        device=device,
    )

    torch.save(model.state_dict(), outdir / "model.pt")
    np.savetxt(outdir / "node_names.txt", np.asarray(taxg.node_names, dtype=object), fmt="%s")

    import pandas as pd

    pd.DataFrame(res["log_rows"]).to_csv(outdir / "training_log.tsv", sep="\t", index=False)
    cfg = {
        "model_type": "hgvae_zi",
        "latent_dim": args.latent_dim,
        "hidden": [args.hidden],
        "dropout": 0.0,
        "activation": "relu",
        "layer_norm": False,
        "input": args.input,
        "taxonomy": args.taxonomy,
        "feature_clades": dataset.sgb_ids,
        "model_kwargs": {
            "rank_vocab": rank_vocab,
            "eps": args.eps,
            "keep_prefixes": bool(args.keep_prefixes),
        },
        "log1p": False,
        "standardize": False,
    }
    with (outdir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    embed_loader = build_hgvae_zi_loader(dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    emb_parts = []
    recon_parts = []
    with torch.no_grad():
        for data in embed_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            emb_parts.append(mu.cpu().numpy())
            xp = model.expected_abundance(data, mu)
            bsz = int(data.batch.max().item()) + 1
            n_nodes = xp.shape[0] // bsz
            leaf_ids = torch.tensor(taxg.leaf_ids, device=xp.device)
            leaf_vals = xp.view(bsz, n_nodes, 1)[:, leaf_ids, 0]
            recon_parts.append(leaf_vals.cpu().numpy())

    emb = np.concatenate(emb_parts, axis=0)
    recon = np.concatenate(recon_parts, axis=0)
    pd.DataFrame(emb, index=sample_ids, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(outdir / "embeddings.tsv", sep="\t")
    pd.DataFrame(recon, index=sample_ids, columns=dataset.sgb_ids).to_csv(outdir / "recon.tsv", sep="\t")


if __name__ == "__main__":
    main()
