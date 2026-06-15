from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from biomevae.data import load_matrix, save_scaler, standardize_train_only, train_val_split, train_val_split_groups
from biomevae.losses import beta_schedule, capacity_schedule, compute_losses
from biomevae.models.flowxformer import FlowXFormerVAE, build_tree_spec
from biomevae.optuna_utils import build_trial_params, load_search_space
from biomevae.taxonomy import load_feature_clades
from biomevae.trainers.train_loop import _resolve_kl_warmup


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-flowxformer")
    ap.add_argument("--input", required=True)
    ap.add_argument("--taxonomy", required=True, help="Path to Phyla.csv (CSV/TSV).")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    ap.add_argument("--activation", type=str, default="leakyrelu", choices=["leakyrelu", "gelu", "relu"])
    ap.add_argument("--layer-norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--distance-bucket-max", type=int, default=8)

    ap.add_argument("--branchlen-mode", choices=["unit", "rank"], default="unit")
    ap.add_argument("--uot", choices=["off", "root_l1"], default="root_l1")
    ap.add_argument("--uot-lambda", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable automatic mixed precision on CUDA (default: enabled when CUDA is available).",
    )

    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    import torch as _torch
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

    ap.add_argument("--consistency-weight", type=float, default=1.0)
    ap.add_argument("--geom-weight", type=float, default=0.01)
    ap.add_argument("--individual-weight", type=float, default=0.1)

    ap.add_argument("--augment-depth-min", type=float, default=0.5)
    ap.add_argument("--augment-depth-max", type=float, default=1.0)
    ap.add_argument("--augment-dropout-min", type=float, default=0.0)
    ap.add_argument("--augment-dropout-max", type=float, default=0.1)

    ap.add_argument("--sample-metadata", type=str, default=None)

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


def _load_metadata(path: str, sample_names: List[str]) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    meta = pd.read_csv(path, sep="\t", dtype=str)
    if "sample_id" not in meta.columns:
        raise SystemExit("sample-metadata must include a 'sample_id' column.")
    meta = meta.set_index("sample_id")
    missing = [s for s in sample_names if s not in meta.index]
    if missing:
        raise SystemExit(f"metadata missing {len(missing)} sample_id entries.")
    meta = meta.reindex(sample_names)
    if "individual_id" in meta.columns:
        individual_ids = meta["individual_id"].astype(str).to_numpy()
    else:
        individual_ids = None
    return meta, individual_ids


def _augment_counts(
    x: torch.Tensor,
    depth_min: float,
    depth_max: float,
    dropout_min: float,
    dropout_max: float,
) -> torch.Tensor:
    batch, feats = x.shape
    out = torch.zeros_like(x)
    fractions = torch.empty((batch, 1), device=x.device).uniform_(depth_min, depth_max)
    totals = x.sum(dim=1)
    for i in range(batch):
        total = int(torch.round(totals[i] * fractions[i]).item())
        if total <= 0:
            continue
        probs = x[i]
        if probs.sum() <= 0:
            continue
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, total, replacement=True)
        out[i].scatter_add_(0, idx, torch.ones(total, device=x.device))
    dropout_p = torch.empty((batch, 1), device=x.device).uniform_(dropout_min, dropout_max)
    mask = torch.rand_like(out) > dropout_p
    out = out * mask
    return out


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable Pearson correlation between two 1-D tensors."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = xc.norm() * yc.norm()
    if denom.item() == 0.0:
        return torch.tensor(0.0, device=x.device)
    return (xc * yc).sum() / (denom + 1e-8)


def _individual_loss(mu: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    if group_ids.numel() == 0:
        return torch.tensor(0.0, device=mu.device)
    valid = group_ids >= 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=mu.device)
    dist_sq = torch.cdist(mu, mu, p=2) ** 2
    same = (group_ids[:, None] == group_ids[None, :]) & valid[:, None] & valid[None, :]
    mask = same & ~torch.eye(mu.size(0), dtype=torch.bool, device=mu.device)
    if not torch.any(mask):
        return torch.tensor(0.0, device=mu.device)
    return dist_sq[mask].mean()


def _build_reference(x_train: np.ndarray, uot_mode: str) -> np.ndarray:
    if uot_mode == "root_l1":
        return x_train.mean(axis=0).astype(np.float32)
    sums = x_train.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return (x_train / sums).mean(axis=0).astype(np.float32)


def _prepare_base_params(args) -> Dict[str, object]:
    return {
        "device": args.device,
        "amp": args.amp if args.amp is not None else args.device.startswith("cuda"),
        "model_type": "flowxformer",
        "latent_dim": args.latent_dim,
        "hidden": list(args.hidden),
        "activation": args.activation,
        "layer_norm": args.layer_norm,
        "dropout": args.dropout,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "distance_bucket_max": args.distance_bucket_max,
        "branchlen_mode": args.branchlen_mode,
        "uot": args.uot,
        "uot_lambda": args.uot_lambda,
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
        "consistency_weight": args.consistency_weight,
        "geom_weight": args.geom_weight,
        "individual_weight": args.individual_weight,
        "augment_depth_min": args.augment_depth_min,
        "augment_depth_max": args.augment_depth_max,
        "augment_dropout_min": args.augment_dropout_min,
        "augment_dropout_max": args.augment_dropout_max,
    }


def _save_optuna_artifacts(outdir: str, params: Dict[str, object], study) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "optuna_best_params.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    try:
        df = study.trials_dataframe()
    except Exception:
        return
    df.to_csv(os.path.join(outdir, "optuna_trials.csv"), index=False)


def _train_flowxformer(
    X_raw: np.ndarray,
    sample_names: List[str],
    feature_clades: List[str],
    taxonomy_path: str,
    outdir: str,
    params: Dict[str, object],
    *,
    metadata: Optional[pd.DataFrame] = None,
    group_ids: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    params = dict(params)
    # Guarantee β saturates before the end of training. ``kl_warmup_frac`` is
    # the canonical knob; absolute ``kl_warmup`` values that would out-run
    # --epochs are clamped by the shared resolver.
    params["kl_warmup"] = _resolve_kl_warmup(params)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if group_ids is not None:
        train_idx, val_idx = train_val_split_groups(len(sample_names), params["val_split"], seed, group_ids)
    else:
        train_idx, val_idx = train_val_split(len(sample_names), params["val_split"], seed)

    X_proc = np.log1p(X_raw).astype(np.float32) if params["log1p"] else X_raw.copy()
    feature_scaler = None
    if params["standardize"]:
        X_proc, feature_scaler = standardize_train_only(X_proc, train_idx)

    tree_spec = build_tree_spec(feature_clades, taxonomy_path, branchlen_mode=params["branchlen_mode"])
    reference = _build_reference(X_raw[train_idx], params["uot"])

    device = torch.device(params["device"])
    use_amp = bool(params.get("amp", False)) and device.type == "cuda"
    model = FlowXFormerVAE(
        input_dim=X_raw.shape[1],
        hidden=list(params["hidden"]),
        latent_dim=params["latent_dim"],
        tree_spec=tree_spec,
        reference=reference,
        d_model=params["d_model"],
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        dropout=params["dropout"],
        activation=params["activation"],
        layer_norm=params["layer_norm"],
        uot_mode=params["uot"],
        uot_lambda=params["uot_lambda"],
        distance_bucket_max=params["distance_bucket_max"],
    ).to(device)

    opt_cls = torch.optim.Adam if params["optimizer"] == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    raw_tensor = torch.from_numpy(X_raw)
    proc_tensor = torch.from_numpy(X_proc)
    if group_ids is not None:
        group_tensor = torch.from_numpy(group_ids.astype(np.int64))
        train_ds = torch.utils.data.TensorDataset(raw_tensor[train_idx], proc_tensor[train_idx], group_tensor[train_idx])
        val_ds = torch.utils.data.TensorDataset(raw_tensor[val_idx], proc_tensor[val_idx], group_tensor[val_idx])
    else:
        train_ds = torch.utils.data.TensorDataset(raw_tensor[train_idx], proc_tensor[train_idx])
        val_ds = torch.utils.data.TensorDataset(raw_tensor[val_idx], proc_tensor[val_idx])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    capacity_end = 0.5 * params["latent_dim"] if params["capacity_end"] is None else params["capacity_end"]

    log_rows = []
    best_val, no_improve = float("inf"), 0
    model_path = os.path.join(outdir, "model.pt")
    if os.path.exists(model_path):
        os.remove(model_path)

    for epoch in range(1, params["epochs"] + 1):
        if params["objective"] == "vanilla":
            beta = 1.0
        else:
            beta = beta_schedule(epoch, params["kl_warmup"], params["beta_max"])
        C = capacity_schedule(epoch, params["capacity_start"], capacity_end, params["capacity_epochs"])

        model.train()
        tr = dict(loss=0.0, recon=0.0, kld=0.0, cons=0.0, geom=0.0, ind=0.0, n=0)
        for batch in train_dl:
            if group_ids is not None:
                xb_raw, xb_proc, xb_group = batch
                xb_group = xb_group.to(device)
            else:
                xb_raw, xb_proc = batch
                xb_group = None
            xb_raw = xb_raw.to(device)
            xb_proc = xb_proc.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                recon, mu, logvar = model(xb_raw)
                loss, r, kl = compute_losses(
                    xb_proc,
                    recon,
                    mu,
                    logvar,
                    recon_kind=params["recon"],
                    huber_delta=params["huber_delta"],
                    objective=params["objective"],
                    beta=beta,
                    free_bits=params["free_bits"],
                    capacity_C=C,
                    capacity_gamma=params["capacity_gamma"],
                )

                if params["uot"] == "root_l1" and params["uot_lambda"] > 0:
                    _, _, root_mismatch = model.featurizer(xb_raw)
                    loss = loss + params["uot_lambda"] * root_mismatch.abs().mean()

                cons_loss = torch.tensor(0.0, device=device)
                if params["consistency_weight"] > 0:
                    x1 = _augment_counts(
                        xb_raw,
                        params["augment_depth_min"],
                        params["augment_depth_max"],
                        params["augment_dropout_min"],
                        params["augment_dropout_max"],
                    )
                    x2 = _augment_counts(
                        xb_raw,
                        params["augment_depth_min"],
                        params["augment_depth_max"],
                        params["augment_dropout_min"],
                        params["augment_dropout_max"],
                    )
                    mu1, _ = model.encode(x1)
                    mu2, _ = model.encode(x2)
                    cons_loss = ((mu1 - mu2) ** 2).mean()
                    loss = loss + params["consistency_weight"] * cons_loss

                geom_loss = torch.tensor(0.0, device=device)
                if params["geom_weight"] > 0 and xb_raw.size(0) > 2:
                    flow_vec = model.flow_vector(xb_raw)
                    flow_dist = torch.pdist(flow_vec, p=2)
                    z_dist = torch.pdist(mu, p=2)
                    if flow_dist.numel() > 0:
                        corr = _pearson_corr(flow_dist, z_dist)
                        geom_loss = 1.0 - corr
                        loss = loss + params["geom_weight"] * geom_loss

                ind_loss = torch.tensor(0.0, device=device)
                if xb_group is not None and params["individual_weight"] > 0:
                    ind_loss = _individual_loss(mu, xb_group)
                    loss = loss + params["individual_weight"] * ind_loss

            grad_scaler.scale(loss).backward()
            if params["grad_clip"] and params["grad_clip"] > 0:
                grad_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["grad_clip"])
            grad_scaler.step(opt)
            grad_scaler.update()

            bsz = xb_raw.size(0)
            tr["loss"] += loss.item() * bsz
            tr["recon"] += r.item() * bsz
            tr["kld"] += kl.item() * bsz
            tr["cons"] += cons_loss.item() * bsz
            tr["geom"] += geom_loss.item() * bsz
            tr["ind"] += ind_loss.item() * bsz
            tr["n"] += bsz

        train_loss = tr["loss"] / tr["n"]
        train_recon = tr["recon"] / tr["n"]
        train_kld = tr["kld"] / tr["n"]
        train_cons = tr["cons"] / tr["n"]
        train_geom = tr["geom"] / tr["n"]
        train_ind = tr["ind"] / tr["n"]

        model.eval()
        vl = dict(loss=0.0, recon=0.0, kld=0.0, n=0)
        with torch.no_grad():
            for batch in val_dl:
                if group_ids is not None:
                    xb_raw, xb_proc, _ = batch
                else:
                    xb_raw, xb_proc = batch
                xb_raw = xb_raw.to(device)
                xb_proc = xb_proc.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    recon, mu, logvar = model(xb_raw)
                    loss, r, kl = compute_losses(
                        xb_proc,
                        recon,
                        mu,
                        logvar,
                        recon_kind=params["recon"],
                        huber_delta=params["huber_delta"],
                        objective=params["objective"],
                        beta=beta,
                        free_bits=params["free_bits"],
                        capacity_C=C,
                        capacity_gamma=params["capacity_gamma"],
                    )
                    if params["uot"] == "root_l1" and params["uot_lambda"] > 0:
                        _, _, root_mismatch = model.featurizer(xb_raw)
                        loss = loss + params["uot_lambda"] * root_mismatch.abs().mean()

                bsz = xb_raw.size(0)
                vl["loss"] += loss.item() * bsz
                vl["recon"] += r.item() * bsz
                vl["kld"] += kl.item() * bsz
                vl["n"] += bsz

        val_loss = vl["loss"] / vl["n"]
        val_recon = vl["recon"] / vl["n"]
        val_kld = vl["kld"] / vl["n"]
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            if verbose:
                print("Stopping early: non-finite train/val loss encountered.")
            break

        log_rows.append({
            "epoch": epoch,
            "objective": params["objective"],
            "beta": beta if params["objective"] in {"beta", "vanilla"} else 0.0,
            "capacity_C": C if params["objective"] == "capacity" else 0.0,
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_kld": train_kld,
            "train_cons": train_cons,
            "train_geom": train_geom,
            "train_ind": train_ind,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kld": val_kld,
        })

        if params["objective"] == "beta":
            aux = f"β={beta:.3f}"
        elif params["objective"] == "vanilla":
            aux = "β=1.000 (vanilla)"
        else:
            aux = f"C={C:.3f} γ={params['capacity_gamma']:.2f}"
        if verbose:
            print(
                f"Epoch {epoch:03d} | {aux} | train={train_loss:.4f} (R={train_recon:.4f},K={train_kld:.4f}) "
                f"| val={val_loss:.4f} (R={val_recon:.4f},K={val_kld:.4f})"
            )

        # Early stopping monitors val_recon, not the β-weighted ELBO. During
        # KL warmup the ELBO is non-stationary (β is still climbing) so a
        # lower val_loss at epoch 1 vs. epoch 50 is not comparable; val_recon
        # is the only scale-invariant signal we have of actual fit quality.
        improved = val_recon + 1e-9 < best_val
        if params["early_stop"] > 0:
            if improved:
                best_val = val_recon
                no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                no_improve += 1
                if no_improve >= params["early_stop"]:
                    if verbose:
                        print("Early stopping.")
                    break
        else:
            if improved:
                best_val = val_recon
            torch.save(model.state_dict(), model_path)

    pd.DataFrame(log_rows).to_csv(os.path.join(outdir, "training_log.tsv"), sep="\t", index=False)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        all_raw = raw_tensor.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            mu, logvar = model.encode(all_raw)
            z = mu.cpu().numpy()
            recon = model(all_raw)[0].cpu().numpy()

    pd.DataFrame(z, index=sample_names, columns=[f"z{i}" for i in range(z.shape[1])]).to_csv(
        os.path.join(outdir, "embeddings.tsv"), sep="\t")
    pd.DataFrame(recon, index=sample_names, columns=[f"f{i}" for i in range(recon.shape[1])]).to_csv(
        os.path.join(outdir, "recon.tsv"), sep="\t")

    cfg: Dict[str, object] = {
        "device": params["device"],
        "amp": params.get("amp", False),
        "model_type": "flowxformer",
        "model_kwargs": {
            "d_model": params["d_model"],
            "n_layers": params["n_layers"],
            "n_heads": params["n_heads"],
            "distance_bucket_max": params["distance_bucket_max"],
            "branchlen_mode": params["branchlen_mode"],
            "uot_mode": params["uot"],
            "uot_lambda": params["uot_lambda"],
        },
        "latent_dim": params["latent_dim"],
        "hidden": list(params["hidden"]),
        "activation": params["activation"],
        "layer_norm": params["layer_norm"],
        "dropout": params["dropout"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "lr": params["lr"],
        "optimizer": params["optimizer"],
        "weight_decay": params["weight_decay"],
        "grad_clip": params["grad_clip"],
        "log1p": params["log1p"],
        "standardize": params["standardize"],
        "val_split": params["val_split"],
        "early_stop": params["early_stop"],
        "objective": params["objective"],
        "recon": params["recon"],
        "huber_delta": params["huber_delta"],
        "kl_warmup": params["kl_warmup"],
        "kl_warmup_frac": params.get("kl_warmup_frac"),
        "beta_max": params["beta_max"],
        "free_bits": params["free_bits"],
        "capacity_start": params["capacity_start"],
        "capacity_end": capacity_end,
        "capacity_epochs": params["capacity_epochs"],
        "capacity_gamma": params["capacity_gamma"],
        "consistency_weight": params["consistency_weight"],
        "geom_weight": params["geom_weight"],
        "individual_weight": params["individual_weight"],
        "augment_depth_min": params["augment_depth_min"],
        "augment_depth_max": params["augment_depth_max"],
        "augment_dropout_min": params["augment_dropout_min"],
        "augment_dropout_max": params["augment_dropout_max"],
        "feature_clades": feature_clades,
        "tree_spec": tree_spec.to_json(),
        "reference": reference.tolist(),
    }
    if metadata is not None:
        cfg["metadata_columns"] = list(metadata.columns)
    cfg.update({"n_samples": len(sample_names), "input_dim": X_raw.shape[1]})

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_scaler(feature_scaler, outdir)

    return {"best_val": best_val}


def _run_optuna(
    args,
    X_raw: np.ndarray,
    sample_names: List[str],
    feature_clades: List[str],
    *,
    metadata: Optional[pd.DataFrame] = None,
    group_ids: Optional[np.ndarray] = None,
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

    study_dir = os.path.join(args.outdir, "optuna_trials")
    os.makedirs(study_dir, exist_ok=True)

    base_params = _prepare_base_params(args)

    def objective(trial):
        params = build_trial_params(trial, base_params, config)
        trial_outdir = os.path.join(study_dir, f"trial_{trial.number:04d}")
        seed = args.seed + trial.number
        trial.set_user_attr("params", params)
        trial.set_user_attr("seed", seed)
        try:
            res = _train_flowxformer(
                X_raw,
                sample_names,
                feature_clades,
                args.taxonomy,
                trial_outdir,
                params,
                metadata=metadata,
                group_ids=group_ids,
                seed=seed,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trial.set_user_attr("oom", True)
            return float("inf")
        if not np.isfinite(res["best_val"]):
            trial.set_user_attr("nonfinite", True)
            return float("inf")
        return res["best_val"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.optuna_trials)

    best_trial = study.best_trial
    if not np.isfinite(best_trial.value):
        raise SystemExit("Optuna did not find any finite validation losses; check input data and search space.")
    best_params = best_trial.user_attrs["params"]
    best_seed = best_trial.user_attrs["seed"]
    res = _train_flowxformer(
        X_raw,
        sample_names,
        feature_clades,
        args.taxonomy,
        args.outdir,
        best_params,
        metadata=metadata,
        group_ids=group_ids,
        seed=best_seed,
        verbose=True,
    )
    _save_optuna_artifacts(args.outdir, best_params, study)
    print(f"\nOptuna best trial #{best_trial.number} | val={best_trial.value:.6f}")
    print(f"\nBest val loss: {res['best_val']:.6f}")


def main() -> None:
    args = build_parser().parse_args()

    X_raw, sample_names = load_matrix(args.input, log1p=False)
    feature_clades = load_feature_clades(args.input)

    metadata = None
    group_ids = None
    if args.sample_metadata:
        metadata, individual_ids = _load_metadata(args.sample_metadata, sample_names)
        if individual_ids is not None:
            _, group_ids = np.unique(individual_ids, return_inverse=True)
        else:
            group_ids = None

    if args.optuna:
        _run_optuna(
            args,
            X_raw,
            sample_names,
            feature_clades,
            metadata=metadata,
            group_ids=group_ids,
        )
        return

    params = _prepare_base_params(args)
    res = _train_flowxformer(
        X_raw,
        sample_names,
        feature_clades,
        args.taxonomy,
        args.outdir,
        params,
        metadata=metadata,
        group_ids=group_ids,
        seed=args.seed,
        verbose=True,
    )
    print(f"\nBest val loss: {res['best_val']:.6f}")


if __name__ == "__main__":
    main()
