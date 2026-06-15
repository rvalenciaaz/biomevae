from typing import Dict, Any, List
import os, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from biomevae.models.vae import VAE
from biomevae.losses import (
    compute_losses,
    beta_schedule,
    capacity_schedule,
    cyclical_beta_schedule,
    effective_number_class_weights,
    focal_ce_balanced,
    gaussian_kl,
    nb_nll,
    zinb_nll,
    supcon_loss,
)
from biomevae.data import train_val_split, standardize_train_only, save_scaler
from biomevae.utils.seeding import (
    _restore_rng_state,
    _snapshot_rng_state,
    set_global_seed,
)


def _resolve_kl_warmup(params: Dict[str, Any], *, verbose: bool = True) -> int:
    """Return the effective absolute KL warmup in epochs.

    Resolution order (no silent modification of user intent):

      1. If ``kl_warmup_frac`` is provided, express it as a fraction of
         ``epochs`` — this is the TreeNB-VAE / HGVAE-ZI convention and is
         the only knob that is guaranteed to saturate β within the run.
      2. Otherwise honour ``kl_warmup`` verbatim in absolute epochs.

    If the resulting warmup is longer than the run, β will still be
    climbing on the final epoch — this is the classical "slow warmup"
    schedule and is often ML-beneficial for β-VAEs because it lets the
    decoder learn to use the latent space before KL pressure kicks in.
    An empirical sweep on the test dataset (``kl_warmup ∈ {25, 50, 300}``
    × ``beta_max ∈ {0.05, 0.3}``) showed that clamping long warmups to
    ``epochs // 4`` drives the encoder into posterior collapse
    (``KL → 0``) for even moderate ``beta_max``. We therefore *warn* the
    caller when the warmup is over-long, but leave the schedule alone so
    reproducibility and ML quality are preserved.
    """
    total_epochs = int(params.get("epochs", 1))

    frac = params.get("kl_warmup_frac")
    if frac is not None:
        resolved = max(1, int(round(total_epochs * float(frac))))
        return resolved

    kl_warmup_abs = int(params.get("kl_warmup", 0) or 0)
    if kl_warmup_abs <= 0:
        return 0

    if kl_warmup_abs >= total_epochs and verbose:
        # Informational only — we do NOT modify the schedule. A long
        # warmup is a legitimate ML choice; the user just needs to know
        # that the plotted ELBO will be non-stationary until the end of
        # training (use val_recon for early stopping and the training
        # curves plot, which is already what biomevae does).
        print(
            f"[kl_warmup] kl_warmup={kl_warmup_abs} is >= epochs={total_epochs}; "
            "β will still be climbing on the final epoch. This is intentional "
            "for slow-warmup schedules — monitor val_recon (not val_loss) for "
            "convergence. Set --kl-warmup-frac for a run-relative schedule."
        )
    return kl_warmup_abs


def _resolve_model(model_type: str):
    if model_type == "euclid":
        return VAE, (lambda kwargs: dict(kwargs))
    if model_type == "hyperbolic":
        from biomevae.models.hyperbolic import HyperbolicVAE

        return HyperbolicVAE, (lambda kwargs: dict(kwargs))
    if model_type == "graph_tax":
        from biomevae.models.graph import TaxonomyGraphVAE, prepare_graph_kwargs

        return TaxonomyGraphVAE, prepare_graph_kwargs
    if model_type == "treeprior":
        from biomevae.models.treeprior import TreeStructuredPriorVAE, prepare_tree_kwargs

        return TreeStructuredPriorVAE, prepare_tree_kwargs
    if model_type == "phylo_fusion":
        from biomevae.models.phylo_fusion import DeepPhyloFusionVAE, prepare_fusion_kwargs

        return DeepPhyloFusionVAE, prepare_fusion_kwargs
    raise ValueError(f"Unknown model_type: {model_type}")

def train_once(
    X: np.ndarray,
    sample_names: List[str],
    outdir: str,
    params: Dict[str, Any],
    seed: int = 42,
    verbose: bool = True,
    return_model: bool = False,
    external_val: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Train a single VAE model.

    Parameters
    ----------
    external_val:
        Optional pre-split validation data.  When provided the internal
        ``val_split`` is skipped and *all* rows of ``X`` are used for
        training while ``external_val`` serves as the early-stopping
        validation set.  This avoids a redundant inner split when the
        caller already holds out data (e.g. Gabriel cross-validation).
    """
    params = copy.deepcopy(params)
    if params.get("epochs", 1) < 1:
        raise ValueError("epochs must be >= 1")
    params.setdefault("objective", "beta")
    if params["objective"] == "vanilla":
        # Vanilla VAE = β-VAE with β_max = 1. Historically we hard-zeroed
        # ``kl_warmup`` and ``free_bits`` here, which forced the model to
        # absorb full unit KL weight from epoch 0 and removed the only
        # safeguard against posterior collapse. We now *default* the
        # vanilla schedule to β_max=1 but honour caller-provided
        # ``kl_warmup`` / ``kl_warmup_frac`` / ``free_bits`` so the
        # scheduler and free-bits both work for the vanilla objective.
        params.setdefault("kl_warmup", 0)
        params["beta_max"] = 1.0
        params.setdefault("free_bits", 0.0)

    # Resolve the effective KL warmup. ``kl_warmup_frac`` (fraction of
    # epochs, matching TreeNB-VAE/HGVAE-ZI) is an opt-in shorthand; an
    # explicit absolute ``kl_warmup`` is preserved verbatim. Over-long
    # warmups (> epochs) are a legitimate slow-warmup schedule and are NOT
    # silently clamped — we only warn, because empirical sweeps show
    # that forcing β to saturate early can induce posterior collapse.
    params["kl_warmup"] = _resolve_kl_warmup(params, verbose=verbose)

    os.makedirs(outdir, exist_ok=True)

    # Seed every RNG (Python, NumPy, PyTorch CPU + CUDA, cuDNN flags,
    # PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG) *before* any model
    # construction or data shuffling happens.  The snapshot/restore
    # pair below ensures that callers that run training trials back to
    # back do not accumulate RNG drift in their own state.
    _rng_snapshot = _snapshot_rng_state()
    set_global_seed(seed)

    n_samples, input_dim = X.shape
    if external_val is not None:
        train_idx = np.arange(n_samples)
        val_idx = np.arange(n_samples)  # placeholder; overridden below
    else:
        train_idx, val_idx = train_val_split(n_samples, params["val_split"], seed)

    scaler = None
    X_proc = X.copy()
    if params["standardize"]:
        X_proc, scaler = standardize_train_only(X_proc, train_idx)

    device = torch.device(params["device"])

    model_type = params.get("model_type", "euclid")
    ModelClass, prep_kwargs = _resolve_model(model_type)
    model_kwargs = params.get("model_kwargs", {})
    if prep_kwargs is not None:
        model_kwargs = prep_kwargs(model_kwargs)

    model = ModelClass(
        input_dim=input_dim,
        hidden=params["hidden"],
        latent_dim=params["latent_dim"],
        dropout=params["dropout"],
        activation=params["activation"],
        layer_norm=params["layer_norm"],
        **model_kwargs,
    ).to(device)

    opt_cls = torch.optim.Adam if params["optimizer"] == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    train_tensor = torch.from_numpy(X_proc)
    if external_val is not None:
        # Use externally provided validation data; all of X is for training.
        val_proc = external_val.copy().astype(np.float32)
        if params["standardize"] and scaler is not None:
            val_proc = (val_proc - scaler["mean"]) / scaler["std"]
        val_tensor = torch.from_numpy(val_proc)
    else:
        val_tensor = train_tensor[val_idx]
        train_tensor = train_tensor[train_idx]

    if model_type == "phylo_fusion" and params["standardize"]:
        phylo_tensor = torch.from_numpy(X)
        if external_val is not None:
            train_phylo = phylo_tensor
            val_phylo = torch.from_numpy(external_val.astype(np.float32))
        else:
            train_phylo, val_phylo = phylo_tensor[train_idx], phylo_tensor[val_idx]
        train_ds = torch.utils.data.TensorDataset(train_tensor, train_phylo)
        val_ds = torch.utils.data.TensorDataset(val_tensor, val_phylo)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
    else:
        train_dl = torch.utils.data.DataLoader(train_tensor, batch_size=params["batch_size"], shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_tensor, batch_size=params["batch_size"], shuffle=False)

    val_array = val_tensor.numpy()
    val_target_mean = float(val_array.mean())
    val_target_total_var = float(np.square(val_array - val_target_mean).sum())
    # Per-feature total variance for per-feature R² (mean-centred per column).
    val_feat_ss_tot = np.sum(np.square(val_array - val_array.mean(axis=0)), axis=0)

    capacity_end = 0.5 * params["latent_dim"] if params["capacity_end"] is None else params["capacity_end"]

    tax_levels = params.get("tax_levels", [])
    tax_As = params.get("tax_As", None)
    lap_L = params.get("lap_L", None)
    lap_weight = float(params.get("lap_weight", 0.0))
    tax_loss_weight = float(params.get("tax_loss_weight", 0.0))

    log_rows = []
    best_val, no_improve = float("inf"), 0
    model_path = os.path.join(outdir, "model.pt")
    # Remove any stale checkpoint that might belong to a previous run with
    # different hyperparameters. Optuna reuses trial directories, so the
    # existing checkpoint could correspond to an incompatible architecture.
    if os.path.exists(model_path):
        os.remove(model_path)

    for epoch in range(1, params["epochs"] + 1):
        # Both "beta" and "vanilla" use the scheduler; vanilla is simply
        # β-VAE with beta_max=1 (enforced in _prepare_params above). The
        # previous branch pinned β=1.0 from epoch 0 regardless of the
        # caller-provided warmup, which drove posterior collapse on
        # weak-signal reconstruction losses (e.g. MSE on log1p compositional
        # data — MetaCardis Vanilla VAE was a textbook example).
        beta = beta_schedule(epoch, params["kl_warmup"], params["beta_max"])
        C = capacity_schedule(epoch, params["capacity_start"], capacity_end, params["capacity_epochs"])

        # ---- Prior blend ramp (for models with a learnable conditional
        # prior, e.g. TreeStructuredPriorVAE).  We linearly ramp the
        # blending coefficient from 0 → 1 over the KL warmup, which pins
        # the early-training KL against a stable N(0, σ²I) reference and
        # only introduces the adaptive prior once the encoder has had a
        # chance to learn an informative posterior.  This is the
        # evidence-based remedy for the posterior collapse observed in
        # treeprior-vae runs.
        if hasattr(model, "set_prior_blend"):
            warmup = max(1, int(params.get("kl_warmup", 0) or 1))
            blend = float(min(1.0, epoch / warmup))
            model.set_prior_blend(blend)

        # ---- Train
        model.train()
        tr = dict(loss=0.0, recon=0.0, kld=0.0, mu_abs=0.0, logvar=0.0, n=0)
        # Count of batches that produced a non-finite loss (NaN/Inf) and had
        # their optimiser step skipped. A single such batch can taint Adam's
        # moment estimates for the rest of training — typical signature is
        # the one-epoch val-ELBO spike seen in the Graph VAE on
        # MetaCardis_2020_a. Any increment here is a signal worth chasing in
        # the training log.
        skipped_batches = 0
        for xb in train_dl:
            phylo_weights = None
            if isinstance(xb, (tuple, list)):
                xb, phylo_weights = xb
                phylo_weights = phylo_weights.to(device)
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            if model_type == "phylo_fusion":
                recon, mu, logvar = model(xb, phylo_weights=phylo_weights)
            else:
                recon, mu, logvar = model(xb)
            prior_info = None
            prior_mu = prior_logvar = None
            if hasattr(model, "conditional_prior"):
                prior_info = model.conditional_prior(xb)
                prior_mu = prior_info.get("mu")
                prior_logvar = prior_info.get("logvar")
            loss, r, kl = compute_losses(
                xb, recon, mu, logvar,
                recon_kind=params["recon"], huber_delta=params["huber_delta"],
                objective=params["objective"], beta=beta, free_bits=params["free_bits"],
                capacity_C=C, capacity_gamma=params["capacity_gamma"],
                prior_mu=prior_mu, prior_logvar=prior_logvar,
            )
            if prior_info is not None and prior_info.get("regularizer") is not None:
                loss = loss + prior_info["regularizer"]

            if tax_As is not None and tax_levels and tax_loss_weight > 0.0:
                from biomevae.losses import compute_losses as _cl
                for lvl in tax_levels:
                    A = tax_As[lvl]
                    # Row-normalise the aggregator so each parent group gets
                    # the *mean* of its children, not the sum. Raw A is a
                    # one-hot indicator (see taxonomy.build_taxonomy_structures);
                    # with family-level groups of size 20–50 the summed values
                    # are orders of magnitude larger than per-feature ones,
                    # which destabilises training (Tax-aware VAE ELBO
                    # oscillated across a full decade on MetaCardis_2020_a).
                    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1.0)
                    A_norm = A / row_sum
                    x_agg = xb @ A_norm.t()
                    recon_agg = recon @ A_norm.t()
                    hier_r = _cl(
                        x_agg, recon_agg, mu, logvar,
                        recon_kind=params["recon"], huber_delta=params["huber_delta"],
                        objective="beta", beta=0.0, free_bits=0.0
                    )[1]
                    loss = loss + tax_loss_weight * hier_r

            if lap_L is not None and lap_weight > 0.0:
                smooth = torch.einsum("bf,fg,bg->", recon, lap_L, recon) / xb.size(0)
                loss = loss + lap_weight * smooth

            # Guard the optimiser step against NaN/Inf losses. Propagating a
            # non-finite loss through ``backward()`` corrupts gradients and,
            # worse, Adam's running moments — the recovery then takes many
            # epochs. We skip the step and continue with the next batch;
            # ``skipped_batches`` is logged per epoch for visibility.
            if not torch.isfinite(loss):
                skipped_batches += 1
                opt.zero_grad(set_to_none=True)
                continue

            loss.backward()
            if params["grad_clip"] and params["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["grad_clip"])
            opt.step()

            bsz = xb.size(0)
            tr["loss"] += loss.item() * bsz
            tr["recon"] += r.item() * bsz
            tr["kld"] += kl.item() * bsz
            tr["mu_abs"] += mu.abs().mean().item() * bsz
            tr["logvar"] += logvar.mean().item() * bsz
            tr["n"] += bsz

        # Guard against division by zero if every batch in an epoch was
        # skipped (pathological but possible if the model explodes globally).
        _n = max(tr["n"], 1)
        train_loss = tr["loss"]/_n; train_recon = tr["recon"]/_n; train_kld = tr["kld"]/_n
        train_mu_abs = tr["mu_abs"]/_n; train_logvar_mean = tr["logvar"]/_n

        # ---- Validate
        model.eval()
        vl = dict(loss=0.0, recon=0.0, kld=0.0, mu_abs=0.0, logvar=0.0, residual=0.0, n=0)
        feat_residual = np.zeros(input_dim, dtype=np.float64)
        # Per-dimension KL for posterior collapse diagnostics.
        kl_per_dim_accum = np.zeros(params["latent_dim"], dtype=np.float64)
        with torch.no_grad():
            for xb in val_dl:
                phylo_weights = None
                if isinstance(xb, (tuple, list)):
                    xb, phylo_weights = xb
                    phylo_weights = phylo_weights.to(device)
                xb = xb.to(device)
                if model_type == "phylo_fusion":
                    recon, mu, logvar = model(xb, phylo_weights=phylo_weights)
                else:
                    recon, mu, logvar = model(xb)
                prior_info = None
                prior_mu = prior_logvar = None
                if hasattr(model, "conditional_prior"):
                    prior_info = model.conditional_prior(xb)
                    prior_mu = prior_info.get("mu")
                    prior_logvar = prior_info.get("logvar")
                loss, r, kl = compute_losses(
                    xb, recon, mu, logvar,
                    recon_kind=params["recon"], huber_delta=params["huber_delta"],
                    objective=params["objective"], beta=beta, free_bits=params["free_bits"],
                    capacity_C=C, capacity_gamma=params["capacity_gamma"],
                    prior_mu=prior_mu, prior_logvar=prior_logvar,
                )
                if prior_info is not None and prior_info.get("regularizer") is not None:
                    loss = loss + prior_info["regularizer"]

                if tax_As is not None and tax_levels and tax_loss_weight > 0.0:
                    from biomevae.losses import compute_losses as _cl
                    for lvl in tax_levels:
                        A = tax_As[lvl]
                        # Same row-normalisation as the training branch: the
                        # aggregated parent receives the mean over children,
                        # not the sum, keeping each level's MSE on the same
                        # scale as the per-feature recon term.
                        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1.0)
                        A_norm = A / row_sum
                        x_agg = xb @ A_norm.t()
                        recon_agg = recon @ A_norm.t()
                        hier_r = _cl(
                            x_agg, recon_agg, mu, logvar,
                            recon_kind=params["recon"], huber_delta=params["huber_delta"],
                            objective="beta", beta=0.0, free_bits=0.0
                        )[1]
                        loss = loss + tax_loss_weight * hier_r

                if lap_L is not None and lap_weight > 0.0:
                    smooth = torch.einsum("bf,fg,bg->", recon, lap_L, recon) / xb.size(0)
                    loss = loss + lap_weight * smooth

                bsz = xb.size(0)
                vl["loss"] += loss.item() * bsz
                vl["recon"] += r.item() * bsz
                vl["kld"] += kl.item() * bsz
                vl["mu_abs"] += mu.abs().mean().item() * bsz
                vl["logvar"] += logvar.mean().item() * bsz
                vl["residual"] += torch.sum((recon - xb) ** 2).item()
                feat_residual += torch.sum((recon - xb) ** 2, dim=0).cpu().numpy()
                # Per-dimension KL, accounting for conditional priors when present.
                if prior_mu is not None and prior_logvar is not None:
                    _pvar = torch.exp(prior_logvar)
                    _diff = mu - prior_mu
                    kl_dim = 0.5 * (
                        (logvar.exp() + _diff.pow(2)) / _pvar - 1.0 + prior_logvar - logvar
                    )
                else:
                    kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (batch, latent)
                kl_per_dim_accum += kl_dim.sum(dim=0).cpu().numpy()
                vl["n"] += bsz

        val_loss = vl["loss"]/vl["n"]; val_recon = vl["recon"]/vl["n"]; val_kld = vl["kld"]/vl["n"]
        val_mu_abs = vl["mu_abs"]/vl["n"]; val_logvar_mean = vl["logvar"]/vl["n"]
        val_residual = vl["residual"]
        val_r2 = float("nan") if val_target_total_var == 0.0 else 1.0 - (val_residual / val_target_total_var)
        with np.errstate(divide="ignore", invalid="ignore"):
            feat_r2 = 1.0 - feat_residual / val_feat_ss_tot
        valid_feat = np.isfinite(feat_r2)
        val_r2_per_feature = float(np.mean(feat_r2[valid_feat])) if np.any(valid_feat) else float("nan")

        # Per-dimension KL diagnostics for posterior collapse detection.
        kl_per_dim = kl_per_dim_accum / max(vl["n"], 1)
        active_units = int((kl_per_dim > 0.01).sum())

        row = {
            "epoch": epoch,
            "objective": params["objective"],
            "beta": beta if params["objective"] in {"beta", "vanilla"} else 0.0,
            "capacity_C": C if params["objective"] == "capacity" else 0.0,
            "train_loss": train_loss, "train_recon": train_recon, "train_kld": train_kld,
            "train_mu_abs": train_mu_abs, "train_logvar_mean": train_logvar_mean,
            "val_loss": val_loss, "val_recon": val_recon, "val_kld": val_kld,
            "val_mu_abs": val_mu_abs, "val_logvar_mean": val_logvar_mean,
            "val_residual": val_residual, "val_target_total_var": val_target_total_var,
            "val_r2": val_r2, "val_r2_per_feature": val_r2_per_feature,
            "active_units": active_units,
            "skipped_batches": skipped_batches,
        }
        for dim_i in range(params["latent_dim"]):
            row[f"kl_dim_{dim_i}"] = float(kl_per_dim[dim_i])
        log_rows.append(row)

        if verbose:
            if params["objective"] == "beta":
                aux = f"β={beta:.3f}"
            elif params["objective"] == "vanilla":
                aux = "β=1.000 (vanilla)"
            else:
                aux = f"C={C:.3f} γ={params['capacity_gamma']:.2f}"
            print(f"Epoch {epoch:03d} | {aux} | train={train_loss:.4f} (R={train_recon:.4f},K={train_kld:.4f}) "
                  f"| val={val_loss:.4f} (R={val_recon:.4f},K={val_kld:.4f}) "
                  f"| |mu|={val_mu_abs:.3f} logvar={val_logvar_mean:.3f} R2={val_r2:.3f} R2f={val_r2_per_feature:.3f}")

        # Monitor reconstruction loss for early stopping instead of the
        # full ELBO (val_loss).  During KL warmup the β weight increases,
        # making val_loss non-comparable across epochs; val_recon is
        # independent of the annealing schedule and therefore stable.
        improved = val_recon + 1e-9 < best_val
        if params["early_stop"] > 0:
            if improved:
                best_val = val_recon; no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                no_improve += 1
                if no_improve >= params["early_stop"]:
                    if verbose: print("Early stopping.")
                    break
        else:
            if improved: best_val = val_recon
            torch.save(model.state_dict(), model_path)

    pd.DataFrame(log_rows).to_csv(os.path.join(outdir, "training_log.tsv"), sep="\t", index=False)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # If the checkpoint was produced by a model with a different
            # architecture (e.g. when reusing Optuna trial directories), just
            # proceed with the in-memory weights from the current run.
            if verbose:
                print(
                    "Warning: checkpoint incompatible with current model; "
                    "using in-memory parameters instead."
                )

    model.eval()
    with torch.no_grad():
        all_tensor = torch.from_numpy(X_proc).to(device)
        if model_type == "phylo_fusion":
            phylo_all = torch.from_numpy(X).to(device)
            mu, logvar = model.encode(all_tensor, phylo_weights=phylo_all)
        else:
            mu, logvar = model.encode(all_tensor)
        z = mu.cpu().numpy()
        recon = model.decoder(mu).cpu().numpy()

    pd.DataFrame(z, index=sample_names, columns=[f"z{i}" for i in range(z.shape[1])]).to_csv(
        os.path.join(outdir, "embeddings.tsv"), sep="\t")
    recon_cols = params.get("feature_clades")
    if not isinstance(recon_cols, list) or len(recon_cols) != recon.shape[1]:
        recon_cols = [f"f{i}" for i in range(recon.shape[1])]
    pd.DataFrame(recon, index=sample_names, columns=recon_cols).to_csv(
        os.path.join(outdir, "recon.tsv"), sep="\t")

    cfg = {k: v for k, v in params.items() if k not in ("tax_As","lap_L")}
    if isinstance(cfg.get("model_kwargs"), dict):
        cfg_mk = dict(cfg["model_kwargs"])
        cfg_mk.pop("phylo_embeddings", None)
        cfg["model_kwargs"] = cfg_mk
    cfg.update({"n_samples": n_samples, "input_dim": input_dim})
    with open(os.path.join(outdir, "config.json"), "w") as f: json.dump(cfg, f, indent=2)
    save_scaler(scaler, outdir)

    # Restore caller RNG state so back-to-back trials don't accumulate drift.
    _restore_rng_state(_rng_snapshot)

    # Final-epoch active units (number of latent dims with KL > 0.01 nats).
    # Consumed by the Optuna search in ``cli/vae_train.py::_run_optuna`` to
    # demote trials that converged via posterior collapse. ``log_rows`` is
    # non-empty at this point because ``epochs >= 1`` is validated on entry.
    final_active_units = int(log_rows[-1]["active_units"]) if log_rows else 0
    return {
        "best_val": best_val,
        "final_val": val_loss,
        "val_recon": val_recon,
        "val_kld": val_kld,
        "active_units": final_active_units,
        "latent_dim": int(params["latent_dim"]),
        "model": model if return_model else None,
        "config": cfg,
    }


# ── PhILR-VAE (compositional) training ────────────────────────────────


def _train_once_philr_family(
    X_leaf: np.ndarray,
    sample_names: List[str],
    outdir: str,
    params: Dict[str, Any],
    taxg: object,
    *,
    model_factory,
    seed: int,
    verbose: bool,
    return_model: bool,
    external_val_leaf: np.ndarray | None,
) -> Dict[str, Any]:
    """Shared training loop for the new compositional PhILR family."""
    params = copy.deepcopy(params)
    os.makedirs(outdir, exist_ok=True)

    _rng_snapshot = _snapshot_rng_state()
    set_global_seed(seed)

    n_samples = X_leaf.shape[0]
    if external_val_leaf is not None:
        train_idx = np.arange(n_samples)
        val_idx = None
    else:
        train_idx, val_idx = train_val_split(
            n_samples, params.get("val_split", 0.1), seed,
        )

    device = torch.device(params.get("device", "cpu"))
    data_kind = params.get("data_kind", "relative")
    likelihood = params.get("likelihood", "philr_gaussian")
    validate_counts = bool(
        params.get("validate_counts", likelihood not in {"philr_gaussian", "dirichlet_tree"})
    )

    X_leaf_t = torch.from_numpy(X_leaf.astype(np.float32))
    if external_val_leaf is not None:
        train_t = X_leaf_t
        val_t = torch.from_numpy(external_val_leaf.astype(np.float32))
    else:
        train_t = X_leaf_t[train_idx]
        val_t = X_leaf_t[val_idx]

    model = model_factory(taxg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    bs = params.get("batch_size", 64)
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_t),
        batch_size=bs, shuffle=True,
    )
    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_t),
        batch_size=bs, shuffle=False,
    )

    epochs = params.get("epochs", 200)
    warmup = max(1, int(epochs * params.get("kl_warmup_frac", 0.25)))
    beta_max_val = params.get("beta_max", 1.0)
    early_stop = params.get("early_stop", 30)
    grad_clip = params.get("grad_clip", 5.0)
    free_bits = float(params.get("free_bits", 0.0))
    concentration_l2 = float(params.get("concentration_l2", 1e-4))

    best_val, no_improve = float("inf"), 0
    model_path = os.path.join(outdir, "model.pt")
    if os.path.exists(model_path):
        os.remove(model_path)

    val_loss = float("nan")
    val_recon_val = float("nan")
    val_kld_val = float("nan")

    for epoch in range(1, epochs + 1):
        beta = beta_schedule(epoch, warmup, beta_max_val)
        model.train()
        for (xb,) in train_dl:
            xb = xb.to(device, non_blocking=True)
            loss, metrics = model.loss(
                xb, likelihood=likelihood, data_kind=data_kind,
                beta=beta, free_bits=free_bits,
                concentration_l2=concentration_l2,
                validate_counts=validate_counts,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        model.eval()
        v_recon = v_kl = v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(device, non_blocking=True)
                loss, metrics = model.loss(
                    xb, likelihood=likelihood, data_kind=data_kind,
                    beta=beta, free_bits=free_bits,
                    concentration_l2=concentration_l2,
                    validate_counts=validate_counts,
                )
                bsz = xb.size(0)
                v_recon += float(metrics["reconstruction_nll"]) * bsz
                v_kl += float(metrics["kl"]) * bsz
                v_loss += float(loss) * bsz
                v_n += bsz

        val_loss = v_loss / v_n
        val_recon_val = v_recon / v_n
        val_kld_val = v_kl / v_n

        if not np.isfinite(val_loss):
            if verbose:
                print("Stopping: non-finite loss.")
            break
        scheduler.step(val_recon_val)

        if verbose:
            print(
                f"Epoch {epoch:03d} | beta={beta:.3f} "
                f"| val={val_loss:.4f} (nll={val_recon_val:.4f} kl={val_kld_val:.4f})"
            )

        improved = val_recon_val + 1e-9 < best_val
        if improved:
            best_val = val_recon_val
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if early_stop > 0 and no_improve >= early_stop:
                if verbose:
                    print("Early stopping.")
                break

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

    model.eval()
    _restore_rng_state(_rng_snapshot)
    return {
        "best_val": best_val,
        "final_val": val_loss,
        "val_recon": val_recon_val,
        "val_kld": val_kld_val,
        "model": model if return_model else None,
        "config": params,
    }


def train_once_philrvae(
    X_leaf: np.ndarray,
    sample_names: List[str],
    outdir: str,
    params: Dict[str, Any],
    taxg: object,
    seed: int = 42,
    verbose: bool = True,
    return_model: bool = False,
    external_val_leaf: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Train the compositional :class:`PhILRVAE`."""
    from biomevae.models.philrvae import PhILRVAE

    def factory(taxg_):
        return PhILRVAE(
            taxg_,
            latent_dim=int(params["latent_dim"]),
            hidden=tuple(params.get("hidden", (256, 128))),
            dropout=float(params.get("dropout", 0.1)),
            count_pseudocount=float(params.get("count_pseudocount", 0.5)),
            relative_pseudocount=float(params.get("relative_pseudocount", 1e-6)),
            default_likelihood=params.get("likelihood", "philr_gaussian"),
            init_coord_scale=float(params.get("init_coord_scale", 0.5)),
            init_concentration=float(params.get("init_concentration", 50.0)),
        )

    return _train_once_philr_family(
        X_leaf, sample_names, outdir, params, taxg,
        model_factory=factory, seed=seed, verbose=verbose,
        return_model=return_model, external_val_leaf=external_val_leaf,
    )


def train_once_hyp_philrvae(
    X_leaf: np.ndarray,
    sample_names: List[str],
    outdir: str,
    params: Dict[str, Any],
    taxg: object,
    seed: int = 42,
    verbose: bool = True,
    return_model: bool = False,
    external_val_leaf: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Train the compositional :class:`HyperbolicPhILRVAE`."""
    from biomevae.models.hyperbolic_philrvae import HyperbolicPhILRVAE

    def factory(taxg_):
        return HyperbolicPhILRVAE(
            taxg_,
            latent_dim=int(params["latent_dim"]),
            curvature=float(params.get("curvature", 1.0)),
            hidden=tuple(params.get("hidden", (256, 128))),
            dropout=float(params.get("dropout", 0.1)),
            count_pseudocount=float(params.get("count_pseudocount", 0.5)),
            relative_pseudocount=float(params.get("relative_pseudocount", 1e-6)),
            default_likelihood=params.get("likelihood", "philr_gaussian"),
            init_coord_scale=float(params.get("init_coord_scale", 0.5)),
            init_concentration=float(params.get("init_concentration", 50.0)),
        )

    return _train_once_philr_family(
        X_leaf, sample_names, outdir, params, taxg,
        model_factory=factory, seed=seed, verbose=verbose,
        return_model=return_model, external_val_leaf=external_val_leaf,
    )


# ── TreeDTM-VAE training ───────────────────────────────────────────────


def train_once_tree_dtm_vae(
    X_nodes: np.ndarray,
    X_leaves: np.ndarray,
    sample_names: List[str],
    outdir: str,
    params: Dict[str, Any],
    topo: object,
    seed: int = 42,
    verbose: bool = True,
    return_model: bool = False,
    external_val_nodes: np.ndarray | None = None,
    external_val_leaves: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Train a :class:`TreeDTMVAE` model.

    Parameters
    ----------
    X_nodes : (n_samples, n_tree_nodes) — full node values (encoder input).
    X_leaves : (n_samples, n_leaves) — leaf values, retained for downstream
        reconstruction comparisons.
    topo : TreeTopology — precomputed tree topology.
    """
    from biomevae.models.tree_dtm_vae import TreeDTMVAE

    params = copy.deepcopy(params)
    os.makedirs(outdir, exist_ok=True)

    _rng_snapshot = _snapshot_rng_state()
    set_global_seed(seed)

    n_samples = X_nodes.shape[0]

    if external_val_nodes is not None:
        train_idx = np.arange(n_samples)
        val_idx = None
    else:
        train_idx, val_idx = train_val_split(
            n_samples, params.get("val_split", 0.1), seed,
        )

    device = torch.device(params.get("device", "cpu"))
    likelihood = params.get("likelihood", "dirichlet_tree_multinomial")

    X_nodes_t = torch.from_numpy(X_nodes.astype(np.float32))
    X_leaves_t = torch.from_numpy(X_leaves.astype(np.float32))

    model = TreeDTMVAE(
        topo,
        hidden=params.get("hidden", 256),
        latent_dim=params.get("latent_dim", 32),
        encoder_layers=params.get("encoder_layers", params.get("n_layers", 2)),
        decoder_hidden=params.get("decoder_hidden", 256),
        decoder_layers=params.get("decoder_layers", 2),
        dropout=params.get("dropout", 0.1),
        encoder_pseudocount=params.get("encoder_pseudocount", 0.5),
        init_concentration=params.get("init_concentration", 50.0),
        likelihood=likelihood,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    if external_val_nodes is not None:
        train_nodes, train_leaves = X_nodes_t, X_leaves_t
        val_nodes = torch.from_numpy(external_val_nodes.astype(np.float32))
        val_leaves = torch.from_numpy(external_val_leaves.astype(np.float32))
    else:
        train_nodes, train_leaves = X_nodes_t[train_idx], X_leaves_t[train_idx]
        val_nodes, val_leaves = X_nodes_t[val_idx], X_leaves_t[val_idx]

    bs = params.get("batch_size", 32)
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_nodes, train_leaves),
        batch_size=bs, shuffle=True,
    )
    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_nodes, val_leaves),
        batch_size=bs, shuffle=False,
    )

    epochs = params.get("epochs", 200)
    kl_warmup_frac = params.get("kl_warmup_frac", 0.25)
    warmup = max(1, int(epochs * kl_warmup_frac))
    beta_max_val = params.get("beta_max", 1.0)
    early_stop = params.get("early_stop", 30)
    grad_clip = params.get("grad_clip", 5.0)
    free_bits = float(params.get("free_bits", 0.0))
    concentration_l2 = float(params.get("concentration_l2", 1e-4))
    validate_counts = bool(params.get("validate_counts", likelihood != "dirichlet_tree"))

    best_val, no_improve = float("inf"), 0
    model_path = os.path.join(outdir, "model.pt")
    if os.path.exists(model_path):
        os.remove(model_path)

    val_loss = float("nan")
    val_recon_val = float("nan")
    val_kld_val = float("nan")

    for epoch in range(1, epochs + 1):
        beta = beta_schedule(epoch, warmup, beta_max_val)

        model.train()
        for x_nodes, _x_leaves in train_dl:
            x_nodes = x_nodes.to(device, non_blocking=True)
            out = model(x_nodes)
            loss, _ = model.loss(
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
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        model.eval()
        v_recon = v_kl = v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for x_nodes, _x_leaves in val_dl:
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

        val_loss = v_loss / v_n
        val_recon_val = v_recon / v_n
        val_kld_val = v_kl / v_n

        if not np.isfinite(val_loss):
            if verbose:
                print("Stopping: non-finite loss.")
            break

        scheduler.step(val_recon_val)

        if verbose:
            print(
                f"Epoch {epoch:03d} | β={beta:.3f} "
                f"| val={val_loss:.4f} (nll={val_recon_val:.4f} "
                f"kl={val_kld_val:.4f})"
            )

        improved = val_recon_val + 1e-9 < best_val
        if early_stop > 0:
            if improved:
                best_val = val_recon_val
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
                best_val = val_recon_val
            torch.save(model.state_dict(), model_path)

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

    model.eval()
    _restore_rng_state(_rng_snapshot)
    return {
        "best_val": best_val,
        "final_val": val_loss,
        "val_recon": val_recon_val,
        "val_kld": val_kld_val,
        "model": model if return_model else None,
        "config": params,
    }


# ── DS-VAE training ────────────────────────────────────────────────────


def _stratified_train_val_split(
    y: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split. Falls back to random split when a
    class has only one sample (rare on small synthetic data)."""
    from sklearn.model_selection import StratifiedShuffleSplit

    y = np.asarray(y)
    n = len(y)
    _uniq, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return train_val_split(n, val_frac, seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(n), y))
    return train_idx, val_idx


def train_once_dsvae(
    X_raw: np.ndarray,
    sample_names: List[str],
    outdir: str,
    params: Dict[str, Any],
    seed: int = 42,
    verbose: bool = True,
    return_model: bool = False,
    external_val: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    external_val_labels: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Train a :class:`DSVAE` model — unsupervised or supervised.

    ``params['supervised']`` selects the variant. In supervised mode
    ``labels`` must be provided as an integer array of shape ``(n,)``
    with values in ``[0, n_classes)``. ``external_val_labels`` mirrors
    ``external_val`` for cross-validation.
    """
    import torch.nn.functional as F
    from biomevae.models.dsvae import DSVAE, philr_mixup
    from biomevae.models.tree_spec import TreeSpec, build_tree_spec

    params = copy.deepcopy(params)
    os.makedirs(outdir, exist_ok=True)

    supervised = bool(params.get("supervised", False))
    _rng_snapshot = _snapshot_rng_state()
    set_global_seed(seed)

    n_samples, input_dim = X_raw.shape

    # --- Tree spec ---
    tree_spec_json = params.get("tree_spec")
    if tree_spec_json and isinstance(tree_spec_json, dict):
        tree_spec = TreeSpec.from_json(tree_spec_json)
    else:
        tree_spec = build_tree_spec(
            params["feature_clades"],
            params["taxonomy_path"],
            branchlen_mode=params.get("branchlen_mode", "unit"),
        )

    # --- Validation split ---
    if supervised and labels is None and external_val_labels is None:
        raise ValueError("DS-VAE supervised=True requires labels.")
    val_idx: np.ndarray
    if external_val is not None:
        train_idx = np.arange(n_samples)
        val_idx = np.arange(n_samples)  # placeholder, overridden below
    elif supervised and labels is not None:
        train_idx, val_idx = _stratified_train_val_split(
            labels, params.get("val_split", 0.1), seed
        )
    else:
        train_idx, val_idx = train_val_split(
            n_samples, params.get("val_split", 0.1), seed,
        )

    # --- Model ---
    device = torch.device(params.get("device", "cpu"))
    n_classes = None
    class_weights_tensor: torch.Tensor | None = None
    if supervised:
        assert labels is not None
        n_classes = int(params.get("n_classes") or int(np.max(labels)) + 1)
        train_labels = labels[train_idx] if external_val is None else labels
        counts = np.bincount(np.asarray(train_labels, dtype=int), minlength=n_classes)
        class_weights_tensor = effective_number_class_weights(
            torch.from_numpy(counts.astype(np.float32)),
            beta=float(params.get("effnum_beta", 0.9999)),
        ).to(device)

    model = DSVAE(
        n_features=input_dim,
        latent_dim=int(params["latent_dim"]),
        tree_spec=tree_spec,
        supervised=supervised,
        n_classes=n_classes,
        hidden=list(params.get("hidden", [512, 256, 128])),
        dropout=float(params.get("dropout", 0.1)),
        pseudocount=float(params.get("pseudocount", 0.5)),
        classifier_hidden=int(params.get("classifier_hidden", 128)),
    ).to(device)

    # --- Optimiser ---
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("lr", 1.5e-3)),
        weight_decay=float(params.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    # --- Data tensors ---
    raw_t = torch.from_numpy(X_raw.astype(np.float32))
    if external_val is not None:
        train_tensor = raw_t
        val_tensor = torch.from_numpy(external_val.astype(np.float32))
        train_y_np = labels if labels is not None else None
        val_y_np = external_val_labels if external_val_labels is not None else None
    else:
        train_tensor = raw_t[train_idx]
        val_tensor = raw_t[val_idx]
        train_y_np = labels[train_idx] if supervised else None
        val_y_np = labels[val_idx] if supervised else None

    bs = int(params.get("batch_size", 64))

    if supervised:
        train_y_t = torch.from_numpy(np.asarray(train_y_np, dtype=np.int64))
        val_y_t = torch.from_numpy(np.asarray(val_y_np, dtype=np.int64))
        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor, train_y_t),
            batch_size=bs, shuffle=True, drop_last=False,
        )
        val_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_tensor, val_y_t),
            batch_size=bs, shuffle=False,
        )
    else:
        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor),
            batch_size=bs, shuffle=True, drop_last=False,
        )
        val_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_tensor),
            batch_size=bs, shuffle=False,
        )

    # --- Schedule parameters ---
    epochs = int(params.get("epochs", 300))
    beta_max_val = float(params.get("beta_max", 1.0))
    n_cycles = int(params.get("beta_n_cycles", 4))
    cycle_len = int(params.get("beta_cycle_len", 50))
    ramp_frac = float(params.get("beta_ramp_frac", 0.5))
    free_bits = float(params.get("free_bits", 0.03))
    grad_clip = float(params.get("grad_clip", 1.0))
    early_stop = int(params.get("early_stop", 50))

    gamma_cls = float(params.get("gamma_cls", 1.0))
    gamma_con = float(params.get("gamma_con", 0.3))
    focal_gamma = float(params.get("focal_gamma", 2.0))
    supcon_tau = float(params.get("supcon_tau", 0.1))
    mixup_alpha = float(params.get("mixup_alpha", 0.2))

    best_val = float("inf")
    no_improve = 0
    model_path = os.path.join(outdir, "model.pt")
    if os.path.exists(model_path):
        os.remove(model_path)

    log_rows: list[dict] = []
    val_loss = float("nan")
    val_recon_val = float("nan")
    val_kld_val = float("nan")
    val_macro_f1 = float("nan")
    val_bacc = float("nan")

    for epoch in range(1, epochs + 1):
        beta = cyclical_beta_schedule(
            epoch,
            n_cycles=n_cycles,
            cycle_len=cycle_len,
            beta_max=beta_max_val,
            ramp_frac=ramp_frac,
        )

        # ---- Train ----
        model.train()
        t_nll = t_kl = t_loss = t_cls = t_con = 0.0
        t_n = 0
        for batch in train_dl:
            if supervised:
                xb, yb = batch
                yb = yb.to(device, non_blocking=True)
            else:
                (xb,) = batch
                yb = None
            xb = xb.to(device, non_blocking=True)

            coords = model.philr(xb)
            y_soft = None
            if supervised and mixup_alpha > 0.0 and xb.size(0) >= 2:
                y_oh = F.one_hot(yb, num_classes=n_classes).float()
                coords_mix, y_soft, _lam = philr_mixup(coords, y_oh, alpha=mixup_alpha)
                mu_z, logvar_z = model.encode_from_coords(coords_mix)
            else:
                mu_z, logvar_z = model.encode_from_coords(coords)

            z = model.reparam(mu_z, logvar_z)
            lib = xb.sum(dim=1, keepdim=True).clamp(min=1.0)
            mu_x = model.decode(z, lib)
            nll = nb_nll(xb, mu_x, model.log_theta)

            if supervised:
                prior_mu, prior_logvar = model.class_prior(yb)
            else:
                prior_mu = torch.zeros_like(mu_z)
                prior_logvar = torch.zeros_like(logvar_z)
            kl_per = gaussian_kl(
                mu_z, logvar_z, prior_mu, prior_logvar, free_bits=free_bits
            )
            kl = kl_per.mean()

            loss = nll + beta * kl
            cls_loss_val = con_loss_val = 0.0

            if supervised:
                logits = model.classify(mu_z)
                if y_soft is not None:
                    cls_loss = focal_ce_balanced(
                        logits, y_soft, gamma=focal_gamma,
                        class_weight=class_weights_tensor,
                    )
                else:
                    cls_loss = focal_ce_balanced(
                        logits, yb, gamma=focal_gamma,
                        class_weight=class_weights_tensor,
                    )
                loss = loss + gamma_cls * cls_loss
                cls_loss_val = float(cls_loss.item())

                if gamma_con > 0.0 and xb.size(0) >= 2:
                    feats = F.normalize(mu_z, dim=-1)
                    con = supcon_loss(feats, yb, temperature=supcon_tau)
                    loss = loss + gamma_con * con
                    con_loss_val = float(con.item())

            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bsz = xb.size(0)
            t_nll += float(nll.item()) * bsz
            t_kl += float(kl.item()) * bsz
            t_loss += float(loss.item()) * bsz
            t_cls += cls_loss_val * bsz
            t_con += con_loss_val * bsz
            t_n += bsz

        _n = max(t_n, 1)
        train_loss = t_loss / _n
        train_recon = t_nll / _n
        train_kld = t_kl / _n
        train_cls = t_cls / _n
        train_con = t_con / _n

        # ---- Validate ----
        model.eval()
        v_nll = v_kl = v_loss = 0.0
        v_n = 0
        all_y_true: list[int] = []
        all_y_pred: list[int] = []
        kl_per_dim_accum = np.zeros(int(params["latent_dim"]), dtype=np.float64)
        with torch.no_grad():
            for batch in val_dl:
                if supervised:
                    xb, yb = batch
                    yb = yb.to(device, non_blocking=True)
                else:
                    (xb,) = batch
                    yb = None
                xb = xb.to(device, non_blocking=True)

                mu_z, logvar_z = model.encode(xb)
                lib = xb.sum(dim=1, keepdim=True).clamp(min=1.0)
                mu_x = model.decode(mu_z, lib)
                nll = nb_nll(xb, mu_x, model.log_theta)

                if supervised:
                    prior_mu, prior_logvar = model.class_prior(yb)
                else:
                    prior_mu = torch.zeros_like(mu_z)
                    prior_logvar = torch.zeros_like(logvar_z)
                kl_per = gaussian_kl(
                    mu_z, logvar_z, prior_mu, prior_logvar, free_bits=0.0
                )
                kl = kl_per.mean()
                loss = nll + beta * kl

                if supervised:
                    logits = model.classify(mu_z)
                    pred = logits.argmax(dim=-1)
                    all_y_true.extend(yb.cpu().numpy().tolist())
                    all_y_pred.extend(pred.cpu().numpy().tolist())

                bsz = xb.size(0)
                v_nll += float(nll.item()) * bsz
                v_kl += float(kl.item()) * bsz
                v_loss += float(loss.item()) * bsz
                v_n += bsz

                pvar = torch.exp(prior_logvar)
                diff = mu_z - prior_mu
                kl_dim = 0.5 * (
                    (logvar_z.exp() + diff.pow(2)) / pvar - 1.0 + prior_logvar - logvar_z
                )
                kl_per_dim_accum += kl_dim.sum(dim=0).cpu().numpy()

        val_loss = v_loss / max(v_n, 1)
        val_recon_val = v_nll / max(v_n, 1)
        val_kld_val = v_kl / max(v_n, 1)
        kl_per_dim = kl_per_dim_accum / max(v_n, 1)
        active_units = int((kl_per_dim > 0.01).sum())

        if supervised and all_y_true:
            from sklearn.metrics import balanced_accuracy_score, f1_score
            val_macro_f1 = float(
                f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
            )
            val_bacc = float(balanced_accuracy_score(all_y_true, all_y_pred))
        else:
            val_macro_f1 = float("nan")
            val_bacc = float("nan")

        if not np.isfinite(val_loss):
            if verbose:
                print("Stopping: non-finite loss.")
            break

        scheduler.step(val_recon_val)

        row = {
            "epoch": epoch,
            "beta": float(beta),
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_nll": train_recon,
            "train_kl": train_kld,
            "train_kld": train_kld,
            "train_cls": train_cls,
            "train_supcon": train_con,
            "val_loss": val_loss,
            "val_recon": val_recon_val,
            "val_nll": val_recon_val,
            "val_kl": val_kld_val,
            "val_kld": val_kld_val,
            "active_units": active_units,
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy": val_bacc,
        }
        for dim_i in range(int(params["latent_dim"])):
            row[f"kl_dim_{dim_i}"] = float(kl_per_dim[dim_i])
        log_rows.append(row)

        if verbose:
            if supervised:
                print(
                    f"ep {epoch:03d} | β={beta:.3f} "
                    f"| train={train_loss:.2f} (nll={train_recon:.2f} "
                    f"kl={train_kld:.2f} cls={train_cls:.3f} "
                    f"con={train_con:.3f}) "
                    f"| val nll={val_recon_val:.2f} F1={val_macro_f1:.3f} "
                    f"BAcc={val_bacc:.3f} AU={active_units}"
                )
            else:
                print(
                    f"ep {epoch:03d} | β={beta:.3f} "
                    f"| train={train_loss:.2f} (nll={train_recon:.2f} "
                    f"kl={train_kld:.2f}) "
                    f"| val nll={val_recon_val:.2f} AU={active_units}"
                )

        # Early stopping: supervised → higher-is-better macro-F1 (negated
        # for the uniform < comparison); unsupervised → val_recon.
        if supervised:
            criterion = -val_macro_f1 if np.isfinite(val_macro_f1) else val_recon_val
        else:
            criterion = val_recon_val
        improved = criterion + 1e-9 < best_val
        if early_stop > 0:
            if improved:
                best_val = criterion
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
                best_val = criterion
            torch.save(model.state_dict(), model_path)

    pd.DataFrame(log_rows).to_csv(
        os.path.join(outdir, "training_log.tsv"), sep="\t", index=False,
    )

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

    model.eval()
    _restore_rng_state(_rng_snapshot)
    return {
        "best_val": best_val,
        "final_val": val_loss,
        "val_recon": val_recon_val,
        "val_kld": val_kld_val,
        "val_macro_f1": val_macro_f1,
        "val_balanced_accuracy": val_bacc,
        "model": model if return_model else None,
        "config": params,
    }
