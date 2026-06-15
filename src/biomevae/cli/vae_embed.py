import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from biomevae.data import load_matrix
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
            raise SystemExit("biomevae-embed: --taxonomy is required for graph_tax models.")
        from biomevae.models.graph import TaxonomyGraphVAE, prepare_graph_kwargs

        mode = kwargs.get("graph_mode", "unweighted")
        graph_spec = build_taxonomy_graph_from_taxonomy(feature_clades, taxonomy_path, mode=mode)
        kwargs = prepare_graph_kwargs({**kwargs, "graph_spec": graph_spec})
        M = TaxonomyGraphVAE
    elif model_type == "treeprior":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-embed: --taxonomy is required for treeprior models.")
        from biomevae.models.treeprior import TreeStructuredPriorVAE, prepare_tree_kwargs

        mode = kwargs.get("graph_mode", "unweighted")
        graph_spec = build_taxonomy_graph_from_taxonomy(feature_clades, taxonomy_path, mode=mode)
        kwargs = prepare_tree_kwargs({**kwargs, "graph_spec": graph_spec})
        M = TreeStructuredPriorVAE
    elif model_type == "phylo_fusion":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-embed: --taxonomy is required for phylo_fusion models.")
        from biomevae.models.phylo_fusion import DeepPhyloFusionVAE, prepare_fusion_kwargs

        method = kwargs.get("phylo_method", "pca")
        dim = int(kwargs.get("phylo_dim", 32))
        phylo = build_phylo_embeddings(feature_clades, taxonomy_path, method=method, dim=dim)
        kwargs = prepare_fusion_kwargs({**kwargs, "phylo_embeddings": phylo})
        M = DeepPhyloFusionVAE
    elif model_type == "hgvae_zi":
        if taxonomy_path is None:
            raise SystemExit("biomevae-embed: --taxonomy is required for hgvae_zi models.")
        from biomevae.models.hgvae_zi import HGVAE_ZI

        rank_vocab = int(kwargs.get("rank_vocab", 0))
        if rank_vocab <= 0:
            raise SystemExit("biomevae-embed: hgvae_zi config missing valid model_kwargs.rank_vocab.")
        return HGVAE_ZI(hidden=int(cfg["hidden"][0]), latent_dim=cfg["latent_dim"], rank_vocab=rank_vocab)
    elif model_type == "flowxformer":
        if taxonomy_path is None or feature_clades is None:
            raise SystemExit("biomevae-embed: --taxonomy is required for flowxformer models.")
        from biomevae.models.flowxformer import FlowXFormerVAE, build_tree_spec

        branchlen_mode = kwargs.pop("branchlen_mode", "unit")
        tree_spec = build_tree_spec(feature_clades, taxonomy_path, branchlen_mode=branchlen_mode)
        reference = np.asarray(cfg.get("reference", []), dtype=np.float32)
        if reference.size == 0:
            raise SystemExit("biomevae-embed: flowxformer config missing reference vector.")
        kwargs = {**kwargs, "tree_spec": tree_spec, "reference": reference}
        M = FlowXFormerVAE
    elif model_type == "tree-dtm-vae":
        if taxonomy_path is None:
            raise SystemExit(
                "biomevae-embed: --taxonomy is required for tree-dtm-vae models."
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
                f"biomevae-embed: --taxonomy is required for {model_type} models."
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
                "biomevae-embed: --taxonomy is required for dsvae models."
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
        raise SystemExit(f"biomevae-embed: unsupported model_type '{model_type}'.")
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
    ap = argparse.ArgumentParser("biomevae-embed")
    ap.add_argument("--input", required=True, help="TSV with clade_name, NCBI_tax_id, samples...")
    ap.add_argument("--model-dir", required=True, help="Directory containing model.pt & config.json")
    ap.add_argument("--taxonomy", default=None, help="Path to taxonomy file required for graph/tree models")
    ap.add_argument("--phyla", default=None, help="Alias for --taxonomy (phyla.tsv/CSV)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--emb-space", choices=["mu", "ball"], default="mu",
                    help="mu: encoder mean (Euclidean or tangent). ball: expmap(mu) for hyperbolic.")
    ap.add_argument("--export-recon", action="store_true", help="Also save recon.tsv")
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
    # (sgb_table row ordering) — skip the consistency check for those.
    expected_clades = cfg.get("feature_clades")
    _tree_orderings = {"tree-dtm-vae"}
    if (
        expected_clades
        and cfg.get("model_type", "euclid") not in _tree_orderings
        and list(expected_clades) != list(feature_clades)
    ):
        raise SystemExit("Input feature ordering does not match the trained model.")

    X, sample_names = load_matrix(args.input, log1p=False)

    if cfg.get("model_type", "euclid") == "hgvae_zi":
        if taxonomy_path is None:
            raise SystemExit("biomevae-embed: --taxonomy is required for hgvae_zi models.")
        from biomevae.models.hgvae_zi import build_hgvae_zi_dataset, build_hgvae_zi_loader

        taxg, dataset, sample_names = build_hgvae_zi_dataset(
            Path(args.input), Path(taxonomy_path),
            eps=float(cfg.get("model_kwargs", {}).get("eps", 1e-6)),
            keep_prefixes=bool(cfg.get("model_kwargs", {}).get("keep_prefixes", False)),
        )
        device = torch.device(args.device)
        model = _build_model(cfg, input_dim=X.shape[1], taxonomy_path=taxonomy_path, feature_clades=feature_clades).to(device)
        state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        loader = build_hgvae_zi_loader(dataset, batch_size=128, shuffle=False)
        emb_parts, recon_parts = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                mu, _ = model.encode(data)
                emb_parts.append(mu.cpu().numpy())
                if args.export_recon:
                    xp = model.expected_abundance(data, mu)
                    bsz = int(data.batch.max().item()) + 1
                    n_nodes = xp.shape[0] // bsz
                    leaf_ids = torch.tensor(taxg.leaf_ids, device=xp.device)
                    leaf_vals = xp.view(bsz, n_nodes, 1)[:, leaf_ids, 0]
                    recon_parts.append(leaf_vals.cpu().numpy())
        emb = np.concatenate(emb_parts, axis=0)
        recon = np.concatenate(recon_parts, axis=0) if recon_parts else None
        pd.DataFrame(emb, index=sample_names, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(
            os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
        if recon is not None:
            pd.DataFrame(recon, index=sample_names, columns=dataset.sgb_ids).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved embeddings.tsv", "and recon.tsv" if recon is not None else "")
        return

    if cfg.get("model_type", "euclid") == "tree-dtm-vae":
        if taxonomy_path is None:
            raise SystemExit(
                "biomevae-embed: --taxonomy is required for tree-dtm-vae models."
            )
        from biomevae.models.tree_dtm_vae import build_treevae_dataset

        _, topo, X_nodes, X_leaves, sample_names, leaf_names, _ = build_treevae_dataset(
            Path(args.input), Path(taxonomy_path),
            data_kind=cfg.get("data_kind", "relative"),
            keep_prefixes=bool(cfg.get("model_kwargs", {}).get("keep_prefixes", False)),
            taxonomy_has_header=bool(
                cfg.get("model_kwargs", {}).get("taxonomy_has_header", False)
            ),
            allow_missing_leaves=True,
        )
        device = torch.device(args.device)
        model = _build_model(cfg, input_dim=X.shape[1], taxonomy_path=taxonomy_path, feature_clades=feature_clades).to(device)
        state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        ds = torch.utils.data.TensorDataset(X_nodes, X_leaves)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        emb_parts, recon_parts = [], []
        with torch.no_grad():
            for x_nodes, x_leaves in loader:
                x_nodes = x_nodes.to(device, non_blocking=True)
                mu, _ = model.encode(x_nodes)
                emb_parts.append(mu.cpu().numpy())
                if args.export_recon:
                    leaf_prob = model.decode(mu)["leaf_prob"]
                    lib = x_leaves.to(device).sum(dim=1, keepdim=True).clamp(min=1.0)
                    recon_parts.append((leaf_prob * lib).cpu().numpy())
        emb = np.concatenate(emb_parts, axis=0)
        recon = np.concatenate(recon_parts, axis=0) if recon_parts else None
        feat_cols = leaf_names if leaf_names else [f"f{i}" for i in range(recon.shape[1] if recon is not None else 0)]
        pd.DataFrame(emb, index=sample_names, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(
            os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
        if recon is not None:
            pd.DataFrame(recon, index=sample_names, columns=feat_cols).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved embeddings.tsv", "and recon.tsv" if recon is not None else "")
        return

    if cfg.get("model_type", "euclid") in ("philrvae", "hyperbolic-philrvae"):
        model_type_emb = cfg["model_type"]
        if taxonomy_path is None:
            raise SystemExit(
                f"biomevae-embed: --taxonomy is required for {model_type_emb} models."
            )
        from biomevae.models.philrvae import build_philrvae_dataset
        data_kind = cfg.get("data_kind", "relative")
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
        model = _build_model(cfg, input_dim=len(leaf_names), taxonomy_path=taxonomy_path, feature_clades=feature_clades).to(device)
        state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        emb_parts, recon_parts = [], []
        ds = torch.utils.data.TensorDataset(X_leaf)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=True)
                mu, _ = model.encode(xb, data_kind=data_kind)
                emb_parts.append(mu.cpu().numpy())
                if args.export_recon:
                    if model_type_emb == "hyperbolic-philrvae":
                        z = model.manifold.projx(model.manifold.expmap0(mu))
                    else:
                        z = mu
                    dec = model.decode(z)
                    lib = xb.sum(dim=1, keepdim=True).clamp(min=1.0)
                    recon_parts.append((dec["leaf_prob"] * lib).cpu().numpy())
        emb = np.concatenate(emb_parts, axis=0)
        recon = np.concatenate(recon_parts, axis=0) if recon_parts else None
        pd.DataFrame(emb, index=sample_names, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(
            os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
        if recon is not None:
            pd.DataFrame(recon, index=sample_names, columns=leaf_names).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved embeddings.tsv", "and recon.tsv" if recon is not None else "")
        return

    if cfg.get("model_type", "euclid") == "dsvae":
        # DSVAE retains the legacy NB-on-PhILR contract (TreeSpec, library_size)
        device = torch.device(args.device)
        model = _build_model(cfg, input_dim=X.shape[1], taxonomy_path=taxonomy_path, feature_clades=feature_clades).to(device)
        state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X.astype(np.float32)).to(device)
            mu, _ = model.encode(xt)
            emb = mu.cpu().numpy()
            recon = None
            if args.export_recon:
                lib = xt.sum(dim=1, keepdim=True).clamp(min=1.0)
                dec_out = model.decode(mu, lib)
                recon_t = dec_out[0] if isinstance(dec_out, tuple) else dec_out
                recon = recon_t.cpu().numpy()
        pd.DataFrame(emb, index=sample_names, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(
            os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
        if recon is not None:
            pd.DataFrame(recon, index=sample_names, columns=[f"f{i}" for i in range(recon.shape[1])]).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved embeddings.tsv", "and recon.tsv" if recon is not None else "")
        return

    if cfg.get("model_type", "euclid") == "capda-vae":
        if taxonomy_path is None:
            raise SystemExit(
                "biomevae-embed: --taxonomy is required for capda-vae models."
            )
        from biomevae.models.capda_vae import (
            _embedding_columns, build_capda_from_config, build_vae_input,
            capda_scale, load_lineage_table, vae_class_probs,
        )

        taxonomy = load_lineage_table(
            taxonomy_path, has_header=bool(cfg.get("taxonomy_has_header", False)))
        levels = tuple(cfg.get("agg_levels"))
        transform = str(cfg.get("transform", "clr"))
        Xin = build_vae_input(X, feature_clades, taxonomy, transform, levels)
        if Xin.shape[1] != int(cfg["input_dim"]):
            raise SystemExit(
                "biomevae-embed: capda-vae input dim mismatch "
                f"(got {Xin.shape[1]} vs trained {cfg['input_dim']})."
            )
        device = torch.device(args.device)
        model = build_capda_from_config(cfg).to(device)
        state = torch.load(os.path.join(args.model_dir, "model.pt"),
                           map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        n_species = int(cfg["n_species"])
        n_classes = int(cfg["n_classes"])
        cols = _embedding_columns(n_species, n_classes)
        # Honest fallback for samples without a stored OOF row (e.g. fresh
        # data): the final VAE's class probabilities, exactly as the LOSO
        # encode step does. Stored leak-free OOF rows always take precedence.
        Xin_s = capda_scale(Xin, cfg)
        probs = vae_class_probs(model, Xin_s, args.device)
        species = np.log1p(X).astype(np.float32)
        emb_arr = np.concatenate([species, probs], axis=1).astype(np.float32)
        # Prefer the stored leak-free OOF rows; the final-VAE probabilities
        # above are only a fallback for samples absent from the OOF table.
        oof_path = os.path.join(args.model_dir, "oof_embeddings.tsv")
        if os.path.exists(oof_path):
            stored = pd.read_csv(oof_path, sep="\t", index_col=0)
            stored = stored.reindex(index=sample_names, columns=cols)
            present = stored.notna().all(axis=1).to_numpy()
            if present.any():
                emb_arr[present] = stored.to_numpy(dtype=np.float32)[present]
        emb_df = pd.DataFrame(emb_arr, index=sample_names, columns=cols)
        emb_df.to_csv(os.path.join(args.outdir, "embeddings.tsv"), sep="\t")

        if args.export_recon:
            with torch.no_grad():
                recon = model(torch.from_numpy(Xin_s).to(device))[0].cpu().numpy()
            recon_species = recon[:, :n_species]
            recon_cols = (
                feature_clades if len(feature_clades) == n_species
                else [f"f{i}" for i in range(n_species)]
            )
            pd.DataFrame(recon_species, index=sample_names, columns=recon_cols).to_csv(
                os.path.join(args.outdir, "recon.tsv"), sep="\t")
        print("Saved embeddings.tsv", "and recon.tsv" if args.export_recon else "")
        return

    X_in = np.log1p(X).astype(np.float32) if cfg.get("log1p", False) else X.astype(np.float32)
    phylo_weights = X_in
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
        model_type = cfg.get("model_type", "euclid")
        if model_type == "flowxformer":
            xt_raw = torch.from_numpy(X.astype(np.float32)).to(device)
            mu, logvar = model.encode(xt_raw)
        else:
            xt = torch.from_numpy(X_in).to(device)
            if model_type == "phylo_fusion" and cfg.get("standardize", False):
                phylo_t = torch.from_numpy(phylo_weights).to(device)
                mu, logvar = model.encode(xt, phylo_weights=phylo_t)
            else:
                mu, logvar = model.encode(xt)
        if args.emb_space == "ball" and getattr(model, "manifold", None) is not None:
            z = model.manifold.expmap0(mu)
            z = model.manifold.projx(z)
            emb = z.cpu().numpy()
        else:
            emb = mu.cpu().numpy()
        if args.export_recon:
            if model_type == "flowxformer":
                recon = model(xt_raw)[0].cpu().numpy()
            elif model_type == "phylo_fusion" and cfg.get("standardize", False):
                recon = model(xt, phylo_weights=phylo_t)[0].cpu().numpy()
            else:
                recon = model(xt)[0].cpu().numpy()
        else:
            recon = None

    pd.DataFrame(emb, index=sample_names, columns=[f"z{i}" for i in range(emb.shape[1])]).to_csv(
        os.path.join(args.outdir, "embeddings.tsv"), sep="\t")
    if recon is not None:
        recon_cols = feature_clades if len(feature_clades) == recon.shape[1] else [f"f{i}" for i in range(recon.shape[1])]
        pd.DataFrame(recon, index=sample_names, columns=recon_cols).to_csv(
            os.path.join(args.outdir, "recon.tsv"), sep="\t")
    print("Saved embeddings.tsv", "and recon.tsv" if recon is not None else "")

if __name__ == "__main__":
    main()
