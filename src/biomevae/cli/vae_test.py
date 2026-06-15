import argparse, os, json
from pathlib import Path
import numpy as np
import torch
from biomevae.data import load_matrix
from biomevae.losses import reconstruction_loss, kl_per_sample
from biomevae.taxonomy import (
    build_phylo_embeddings,
    build_taxonomy_graph_from_taxonomy,
    load_feature_clades,
)

def _load_config(model_dir: str):
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        return json.load(f)

def _load_scaler(model_dir: str):
    path = os.path.join(model_dir, "feature_scaler.npz")
    if os.path.exists(path):
        arr = np.load(path)
        return {"mean": arr["mean"], "std": arr["std"]}
    return None

def _apply_scaler(X: np.ndarray, scaler):
    if scaler is None:
        return X
    mean = scaler["mean"]; std = scaler["std"]
    std = std.copy(); std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)

def _build_model(cfg, input_dim, taxonomy_path=None, feature_clades=None):
    model_type = cfg.get("model_type", "euclid")
    kwargs = dict(cfg.get("model_kwargs", {}))
    kwargs.pop("graph_spec", None)
    if "phylo_embeddings" in kwargs:
        kwargs.pop("phylo_embeddings", None)
    if model_type == "hyperbolic":
        from biomevae.models.hyperbolic import HyperbolicVAE as M
    elif model_type == "graph_tax":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-test: --taxonomy is required for graph_tax models.")
        from biomevae.models.graph import TaxonomyGraphVAE, prepare_graph_kwargs

        mode = kwargs.get("graph_mode", "unweighted")
        graph_spec = build_taxonomy_graph_from_taxonomy(feature_clades, taxonomy_path, mode=mode)
        kwargs = prepare_graph_kwargs({**kwargs, "graph_spec": graph_spec})
        M = TaxonomyGraphVAE
    elif model_type == "treeprior":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-test: --taxonomy is required for treeprior models.")
        from biomevae.models.treeprior import TreeStructuredPriorVAE, prepare_tree_kwargs

        mode = kwargs.get("graph_mode", "unweighted")
        graph_spec = build_taxonomy_graph_from_taxonomy(feature_clades, taxonomy_path, mode=mode)
        kwargs = prepare_tree_kwargs({**kwargs, "graph_spec": graph_spec})
        M = TreeStructuredPriorVAE
    elif model_type == "phylo_fusion":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-test: --taxonomy is required for phylo_fusion models.")
        from biomevae.models.phylo_fusion import DeepPhyloFusionVAE, prepare_fusion_kwargs

        method = kwargs.get("phylo_method", "pca")
        dim = int(kwargs.get("phylo_dim", 32))
        phylo = build_phylo_embeddings(feature_clades, taxonomy_path, method=method, dim=dim)
        kwargs = prepare_fusion_kwargs({**kwargs, "phylo_embeddings": phylo})
        M = DeepPhyloFusionVAE
    elif model_type == "hgvae_zi":
        from biomevae.models.hgvae_zi import HGVAE_ZI as M

        rank_vocab = int(kwargs.get("rank_vocab", 0))
        if rank_vocab <= 0:
            raise SystemExit("biomevae-test: hgvae_zi config missing valid model_kwargs.rank_vocab.")
        return M(hidden=int(cfg["hidden"][0]), latent_dim=cfg["latent_dim"], rank_vocab=rank_vocab)
    elif model_type == "tree-dtm-vae":
        if taxonomy_path is None:
            raise SystemExit(
                f"biomevae-test: --taxonomy is required for {model_type} models."
            )
        from biomevae.models.taxonomy_tree import build_taxonomy_graph_from_phyla_tsv
        from biomevae.models.tree_dtm_vae import TreeDTMVAE, build_tree_topology

        taxg = build_taxonomy_graph_from_phyla_tsv(
            Path(taxonomy_path),
            keep_prefixes=bool(kwargs.get("keep_prefixes", False)),
            has_header=bool(kwargs.get("taxonomy_has_header", False)),
            on_duplicate_leaf="ignore_same",
        )
        topo = build_tree_topology(taxg)
        return TreeDTMVAE(
            topo,
            hidden=int(cfg.get("hidden", 256)),
            latent_dim=int(cfg["latent_dim"]),
            encoder_layers=int(cfg.get("encoder_layers", 2)),
            decoder_hidden=int(cfg.get("decoder_hidden", 256)),
            decoder_layers=int(cfg.get("decoder_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            encoder_pseudocount=float(cfg.get("encoder_pseudocount", 0.5)),
            init_concentration=float(cfg.get("init_concentration", 50.0)),
            likelihood=cfg.get("likelihood", "dirichlet_tree_multinomial"),
        )
    elif model_type in ("philrvae", "hyperbolic-philrvae"):
        if taxonomy_path is None:
            raise SystemExit(
                f"biomevae-test: --taxonomy is required for {model_type} models."
            )
        from biomevae.models.taxonomy_tree import build_taxonomy_graph_from_phyla_tsv
        taxg = build_taxonomy_graph_from_phyla_tsv(
            Path(taxonomy_path),
            keep_prefixes=bool(kwargs.get("keep_prefixes", False)),
            has_header=bool(kwargs.get("taxonomy_has_header", False)),
            on_duplicate_leaf="ignore_same",
        )
        common = dict(
            latent_dim=int(cfg["latent_dim"]),
            hidden=tuple(cfg.get("hidden", [256, 128])),
            dropout=float(cfg.get("dropout", 0.1)),
            count_pseudocount=float(cfg.get("count_pseudocount", 0.5)),
            relative_pseudocount=float(cfg.get("relative_pseudocount", 1e-6)),
            default_likelihood=cfg.get("likelihood", "philr_gaussian"),
            init_coord_scale=float(cfg.get("init_coord_scale", 0.5)),
            init_concentration=float(cfg.get("init_concentration", 50.0)),
        )
        if model_type == "hyperbolic-philrvae":
            from biomevae.models.hyperbolic_philrvae import HyperbolicPhILRVAE
            return HyperbolicPhILRVAE(
                taxg,
                curvature=float(cfg.get("curvature", 1.0)),
                **common,
            )
        from biomevae.models.philrvae import PhILRVAE
        return PhILRVAE(taxg, **common)
    elif model_type == "dsvae":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit(
                "biomevae-test: --taxonomy is required for dsvae models."
            )
        from biomevae.models.dsvae import DSVAE
        from biomevae.models.tree_spec import TreeSpec, build_tree_spec as _build_ts

        ts_json = cfg.get("tree_spec")
        if ts_json:
            tree_spec = TreeSpec.from_json(ts_json)
        else:
            branchlen = cfg.get("branchlen_mode", "unit")
            tree_spec = _build_ts(feature_clades, taxonomy_path, branchlen_mode=branchlen)
        supervised = bool(cfg.get("supervised", False))
        n_classes = int(cfg.get("n_classes", 0)) if supervised else None
        return DSVAE(
            n_features=input_dim,
            latent_dim=int(cfg["latent_dim"]),
            tree_spec=tree_spec,
            supervised=supervised,
            n_classes=n_classes,
            hidden=list(cfg.get("hidden", [512, 256, 128])),
            dropout=float(cfg.get("dropout", 0.1)),
            pseudocount=float(cfg.get("pseudocount", 0.5)),
            classifier_hidden=int(cfg.get("classifier_hidden", 128)),
        )
    elif model_type == "euclid":
        from biomevae.models.vae import VAE as M
    else:
        raise SystemExit(f"biomevae-test: unsupported model_type '{model_type}'.")
    return M(
        input_dim=input_dim,
        hidden=cfg["hidden"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
        layer_norm=cfg["layer_norm"],
        **kwargs,
    )

def build_parser():
    ap = argparse.ArgumentParser("biomevae-test")
    ap.add_argument("--input", required=True, help="TSV with clade_name, NCBI_tax_id, samples...")
    ap.add_argument("--model-dir", required=True, help="Directory containing model.pt & config.json")
    ap.add_argument("--taxonomy", default=None, help="Path to taxonomy file required for graph/tree models")
    ap.add_argument("--phyla", default=None, help="Alias for --taxonomy (phyla.tsv/CSV)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--export", action="store_true", help="Export recon.tsv and embeddings.tsv as well")
    import torch; ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap

def main():
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    taxonomy_path = args.taxonomy or args.phyla

    cfg = _load_config(args.model_dir)
    feature_clades = load_feature_clades(args.input)
    # Tree-softmax variants (tree-dtm-vae) store leaf names in
    # tree-traversal order, which differs from load_feature_clades
    # (sgb_table row ordering) — skip the check.
    expected_clades = cfg.get("feature_clades")
    _tree_orderings = {"tree-dtm-vae"}
    if (
        expected_clades
        and cfg.get("model_type") not in _tree_orderings
        and list(expected_clades) != list(feature_clades)
    ):
        raise SystemExit("Input feature ordering does not match the trained model.")

    if cfg.get("model_type") == "hgvae_zi":
        if taxonomy_path is None:
            raise SystemExit("biomevae-test: --taxonomy is required for hgvae_zi models.")
        from biomevae.models.hgvae_zi import (
            build_hgvae_zi_dataset,
            build_hgvae_zi_loader,
            zi_lognormal_nll,
        )
        taxg, dataset, sample_names = build_hgvae_zi_dataset(
            Path(args.input), Path(taxonomy_path),
            eps=float(cfg.get("model_kwargs", {}).get("eps", 1e-6)),
            keep_prefixes=bool(cfg.get("model_kwargs", {}).get("keep_prefixes", False)),
        )
        device = torch.device(args.device)
        model = _build_model(cfg, input_dim=len(dataset.sgb_ids), taxonomy_path=taxonomy_path, feature_clades=dataset.sgb_ids).to(device)
        state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        loader = build_hgvae_zi_loader(dataset, batch_size=128, shuffle=False)
        recon_vals = []
        mu_parts = []
        logvar_parts = []
        recon_leaf_parts = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                recon_vals.append(float(zi_lognormal_nll(data.y, out["mu_log"], out["log_sig_log"], out["logit_pi"]).item()))
                mu_parts.append(out["mu"].cpu().numpy())
                logvar_parts.append(out["logvar"].cpu().numpy())
                if args.export:
                    xp = model.expected_abundance(data, out["mu"])
                    bsz = int(data.batch.max().item()) + 1
                    n_nodes = xp.shape[0] // bsz
                    leaf_ids = torch.tensor(taxg.leaf_ids, device=xp.device)
                    recon_leaf_parts.append(xp.view(bsz, n_nodes, 1)[:, leaf_ids, 0].cpu().numpy())
        mu = np.concatenate(mu_parts, axis=0)
        logvar = np.concatenate(logvar_parts, axis=0)
        kl = float(0.5 * np.mean(np.sum(np.exp(logvar) + mu**2 - 1.0 - logvar, axis=1)))
        report = {
            "reconstruction": float(np.mean(recon_vals)),
            "kl_mean": kl,
            "prior_regularizer": 0.0,
            "beta_loss_at_beta_max": float(np.mean(recon_vals) + float(cfg.get("beta_max", 1.0)) * kl),
            "capacity_loss_at_C_end": float(np.mean(recon_vals) + abs(kl - 0.5 * cfg["latent_dim"])),
        }
        with open(os.path.join(args.outdir, "test_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        if args.export:
            import pandas as pd
            pd.DataFrame(mu, index=sample_names, columns=[f"z{i}" for i in range(mu.shape[1])]).to_csv(
                os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
            recon = np.concatenate(recon_leaf_parts, axis=0)
            pd.DataFrame(recon, index=sample_names, columns=dataset.sgb_ids).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved test_report.json", "and embeddings/recon" if args.export else "")
        return

    if cfg.get("model_type") == "tree-dtm-vae":
        if taxonomy_path is None:
            raise SystemExit(
                "biomevae-test: --taxonomy is required for tree-dtm-vae models."
            )
        from biomevae.models.tree_dtm_vae import build_treevae_dataset
        likelihood = cfg.get("likelihood", "dirichlet_tree_multinomial")
        data_kind = cfg.get("data_kind", "relative")
        _, topo, X_nodes, X_leaves, sample_names, leaf_names, _ = build_treevae_dataset(
            Path(args.input), Path(taxonomy_path),
            data_kind=data_kind,
            keep_prefixes=bool(cfg.get("model_kwargs", {}).get("keep_prefixes", False)),
            taxonomy_has_header=bool(
                cfg.get("model_kwargs", {}).get("taxonomy_has_header", False)
            ),
            allow_missing_leaves=True,
        )
        device = torch.device(args.device)
        model = _build_model(
            cfg, input_dim=len(leaf_names),
            taxonomy_path=taxonomy_path, feature_clades=feature_clades,
        ).to(device)
        state = torch.load(
            os.path.join(args.model_dir, "model.pt"),
            map_location=device, weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        ds = torch.utils.data.TensorDataset(X_nodes, X_leaves)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        nll_vals, kl_vals, mu_parts, logvar_parts, recon_parts = [], [], [], [], []
        with torch.no_grad():
            for x_nodes, x_leaves in loader:
                x_nodes = x_nodes.to(device, non_blocking=True)
                out = model(x_nodes)
                _, metrics = model.loss(
                    x_nodes, outputs=out, likelihood=likelihood, beta=1.0,
                    free_bits=0.0, concentration_l2=0.0,
                    validate_counts=likelihood != "dirichlet_tree",
                )
                nll_vals.append(float(metrics["reconstruction_nll"]))
                kl_vals.append(float(metrics["kl"]))
                mu_parts.append(out["mu_z"].cpu().numpy())
                logvar_parts.append(out["logvar_z"].cpu().numpy())
                if args.export:
                    leaf_prob = out["leaf_prob"]
                    lib = x_leaves.to(device).sum(dim=1, keepdim=True).clamp(min=1.0)
                    recon_parts.append((leaf_prob * lib).cpu().numpy())
        mu = np.concatenate(mu_parts, axis=0)
        logvar = np.concatenate(logvar_parts, axis=0)
        kl = float(0.5 * np.mean(np.sum(np.exp(logvar) + mu**2 - 1.0 - logvar, axis=1)))
        r = float(np.mean(nll_vals))
        beta_max = float(cfg.get("beta_max", 1.0))
        report = {
            "reconstruction": r,
            "kl_mean": kl,
            "prior_regularizer": 0.0,
            "beta_loss_at_beta_max": r + beta_max * kl,
            "capacity_loss_at_C_end": r + abs(kl - 0.5 * cfg["latent_dim"]),
        }
        with open(os.path.join(args.outdir, "test_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        if args.export:
            import pandas as pd
            pd.DataFrame(mu, index=sample_names, columns=[f"z{i}" for i in range(mu.shape[1])]).to_csv(
                os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
            recon = np.concatenate(recon_parts, axis=0)
            feat_cols = leaf_names if leaf_names else [f"f{i}" for i in range(recon.shape[1])]
            pd.DataFrame(recon, index=sample_names, columns=feat_cols).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved test_report.json", "and embeddings/recon" if args.export else "")
        return

    if cfg.get("model_type") == "dsvae":
        if taxonomy_path is None:
            raise SystemExit("biomevae-test: --taxonomy is required for dsvae models.")
        from biomevae.losses import nb_nll
        X, sample_names = load_matrix(args.input, log1p=False)
        device = torch.device(args.device)
        model = _build_model(
            cfg, input_dim=X.shape[1],
            taxonomy_path=taxonomy_path, feature_clades=feature_clades,
        ).to(device)
        state = torch.load(
            os.path.join(args.model_dir, "model.pt"),
            map_location=device, weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X.astype(np.float32)).to(device)
            mu_x, mu_z, logvar_z = model(xt)
            r = float(nb_nll(xt, mu_x, model.log_theta).item())
            kl = float(0.5 * (
                mu_z.pow(2) + logvar_z.exp() - 1.0 - logvar_z
            ).sum(dim=-1).mean().item())
        beta_max = float(cfg.get("beta_max", 1.0))
        report = {
            "reconstruction": r,
            "kl_mean": kl,
            "prior_regularizer": 0.0,
            "beta_loss_at_beta_max": r + beta_max * kl,
            "capacity_loss_at_C_end": r + abs(kl - 0.5 * cfg["latent_dim"]),
        }
        with open(os.path.join(args.outdir, "test_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        if args.export:
            import pandas as pd
            emb = mu_z.cpu().numpy()
            pd.DataFrame(
                emb, index=sample_names,
                columns=[f"z{i}" for i in range(emb.shape[1])],
            ).to_csv(os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
            recon = mu_x.cpu().numpy()
            pd.DataFrame(
                recon, index=sample_names,
                columns=[f"f{i}" for i in range(X.shape[1])],
            ).to_csv(os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved test_report.json", "and embeddings/recon" if args.export else "")
        return

    if cfg.get("model_type") in ("philrvae", "hyperbolic-philrvae"):
        _mt = cfg["model_type"]
        if taxonomy_path is None:
            raise SystemExit(
                f"biomevae-test: --taxonomy is required for {_mt} models."
            )
        from biomevae.models.philrvae import build_philrvae_dataset
        data_kind = cfg.get("data_kind", "relative")
        likelihood = cfg.get("likelihood", "philr_gaussian")
        taxg, X_leaf, _X_nodes, sample_names, leaf_names, _ = build_philrvae_dataset(
            Path(args.input), Path(taxonomy_path),
            data_kind=data_kind,
            keep_prefixes=bool(cfg.get("model_kwargs", {}).get("keep_prefixes", False)),
            taxonomy_has_header=bool(
                cfg.get("model_kwargs", {}).get("taxonomy_has_header", False)
            ),
            allow_missing_leaves=True,
        )
        device = torch.device(args.device)
        model = _build_model(
            cfg, input_dim=len(leaf_names),
            taxonomy_path=taxonomy_path, feature_clades=feature_clades,
        ).to(device)
        state = torch.load(
            os.path.join(args.model_dir, "model.pt"),
            map_location=device, weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        nll_vals, kl_vals, mu_parts, recon_parts = [], [], [], []
        ds = torch.utils.data.TensorDataset(X_leaf)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=True)
                out = model(xb, data_kind=data_kind)
                _, metrics = model.loss(
                    xb, out=out, likelihood=likelihood, data_kind=data_kind,
                    beta=1.0, free_bits=0.0, concentration_l2=0.0,
                    validate_counts=likelihood not in {"philr_gaussian", "dirichlet_tree"},
                )
                nll_vals.append(float(metrics["reconstruction_nll"]))
                kl_vals.append(float(metrics["kl"]))
                mu_parts.append(out["mu_z"].cpu().numpy())
                if args.export:
                    lib = xb.sum(dim=1, keepdim=True).clamp(min=1.0)
                    recon_parts.append((out["leaf_prob"] * lib).cpu().numpy())
        mu = np.concatenate(mu_parts, axis=0)
        r = float(np.mean(nll_vals))
        kl = float(np.mean(kl_vals))
        beta_max = float(cfg.get("beta_max", 1.0))
        report = {
            "reconstruction": r,
            "kl_mean": kl,
            "prior_regularizer": 0.0,
            "beta_loss_at_beta_max": r + beta_max * kl,
            "capacity_loss_at_C_end": r + abs(kl - 0.5 * cfg["latent_dim"]),
        }
        with open(os.path.join(args.outdir, "test_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        if args.export:
            import pandas as pd
            pd.DataFrame(mu, index=sample_names, columns=[f"z{i}" for i in range(mu.shape[1])]).to_csv(
                os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
            recon = np.concatenate(recon_parts, axis=0)
            pd.DataFrame(recon, index=sample_names, columns=leaf_names).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved test_report.json", "and embeddings/recon" if args.export else "")
        return

    if cfg.get("model_type") == "capda-vae":
        if taxonomy_path is None:
            raise SystemExit(
                "biomevae-test: --taxonomy is required for capda-vae models."
            )
        from biomevae.models.capda_vae import (
            build_capda_from_config, build_vae_input, capda_scale,
            load_lineage_table,
        )

        X, sample_names = load_matrix(args.input, log1p=False)
        taxonomy = load_lineage_table(
            taxonomy_path, has_header=bool(cfg.get("taxonomy_has_header", False)))
        levels = tuple(cfg.get("agg_levels"))
        transform = str(cfg.get("transform", "clr"))
        Xin = build_vae_input(X, feature_clades, taxonomy, transform, levels)
        Xin_s = capda_scale(Xin, cfg)
        device = torch.device(args.device)
        model = build_capda_from_config(cfg).to(device)
        state = torch.load(os.path.join(args.model_dir, "model.pt"),
                           map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(Xin_s).to(device)
            recon, mu, logvar, _logits, _z = model(xt)
            r = float(((recon - xt) ** 2).mean().item())
            kl = float(0.5 * (
                mu.pow(2) + logvar.exp() - 1.0 - logvar
            ).sum(dim=-1).mean().item())
        latent_dim = int(cfg["latent_dim"])
        beta_max = float(cfg.get("beta_max", 1.0))
        report = {
            "reconstruction": r,
            "kl_mean": kl,
            "prior_regularizer": 0.0,
            "beta_loss_at_beta_max": r + beta_max * kl,
            "capacity_loss_at_C_end": r + abs(kl - 0.5 * latent_dim),
        }
        with open(os.path.join(args.outdir, "test_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        if args.export:
            import pandas as pd
            emb = mu.cpu().numpy()
            pd.DataFrame(
                emb, index=sample_names,
                columns=[f"z{i}" for i in range(emb.shape[1])],
            ).to_csv(os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
            n_species = int(cfg["n_species"])
            recon_species = recon.cpu().numpy()[:, :n_species]
            recon_cols = (
                feature_clades if len(feature_clades) == n_species
                else [f"f{i}" for i in range(n_species)]
            )
            pd.DataFrame(recon_species, index=sample_names, columns=recon_cols).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved test_report.json", "and embeddings/recon" if args.export else "")
        return

    X, sample_names = load_matrix(args.input, log1p=False)

    X_in = np.log1p(X).astype(np.float32) if cfg.get("log1p", False) else X
    scaler = _load_scaler(args.model_dir) if cfg.get("standardize", False) else None
    X_in = _apply_scaler(X_in, scaler)

    device = torch.device(args.device)
    model = _build_model(
        cfg,
        input_dim=X_in.shape[1],
        taxonomy_path=taxonomy_path,
        feature_clades=feature_clades,
    ).to(device)
    state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        xt = torch.from_numpy(X_in).to(device)
        recon, mu, logvar = model(xt)
        prior_info = None
        prior_mu = prior_logvar = None
        if hasattr(model, "conditional_prior"):
            prior_info = model.conditional_prior(xt)
            prior_mu = prior_info.get("mu")
            prior_logvar = prior_info.get("logvar")
        # Use per-feature-mean reduction here so the reported value remains
        # an interpretable per-feature error metric (not the ELBO-scale
        # sum-over-features recon that the training loop now uses).
        r = reconstruction_loss(
            xt,
            recon,
            kind=cfg.get("recon", "mae"),
            huber_delta=cfg.get("huber_delta", 1.0),
            per_feature="mean",
        ).item()
        kl = kl_per_sample(
            mu,
            logvar,
            free_bits=(cfg.get("free_bits",0.0) if cfg.get("objective","beta")=="beta" else 0.0),
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
        ).mean().item()
        reg_term = 0.0
        if prior_info is not None and prior_info.get("regularizer") is not None:
            reg_term = float(prior_info["regularizer"].item())
        beta_max = float(cfg.get("beta_max", 1.0))
        cap_end = (0.5 * cfg["latent_dim"]) if cfg.get("capacity_end") in (None, "None") else float(cfg.get("capacity_end"))
        cap_gamma = float(cfg.get("capacity_gamma", 1.0))
        beta_loss = r + reg_term + beta_max * kl
        cap_loss = r + reg_term + cap_gamma * float(abs(kl - cap_end))
        report = {
            "reconstruction": r,
            "kl_mean": kl,
            "prior_regularizer": reg_term,
            "beta_loss_at_beta_max": beta_loss,
            "capacity_loss_at_C_end": cap_loss,
        }
        with open(os.path.join(args.outdir, "test_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        if args.export:
            import pandas as pd
            emb = mu.cpu().numpy()
            pd.DataFrame(emb, index=sample_names, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(
                os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
            recon_np = recon.cpu().numpy()
            recon_cols = feature_clades if len(feature_clades) == recon_np.shape[1] else [f"f{i}" for i in range(recon_np.shape[1])]
            pd.DataFrame(recon_np, index=sample_names, columns=recon_cols).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")

    print("Saved test_report.json", "and embeddings/recon" if args.export else "")

if __name__ == "__main__":
    main()
