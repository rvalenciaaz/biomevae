"""Encode held-out samples with a strict-LOSO trained model.

The strict-LOSO pipeline trains one encoder per ``(model, held_out_study)``
fold on N-1 studies; this CLI applies that trained encoder to the held-out
cohort so the downstream classifier can be evaluated on samples the
encoder never saw.  Works for every model in the LOSO sweep:

* DIVA backbones (``diva-hyp-philrvae``, ``diva-beta-vae``) —
  rebuilds the model class from ``config.json`` + ``model.pt`` and emits
  ``embeddings.tsv`` plus the per-factor slices (``embeddings_z_d.tsv`` /
  ``embeddings_z_y.tsv`` / ``embeddings_z_x.tsv``) exactly as the trainer
  does.
* Non-DIVA VAEs (``tree-dtm-vae``, ``hyperbolic-philrvae``, ``euclid``) —
  delegates to ``biomevae-embed`` (which already supports those backbones).
* XGBoost passthrough featurisers (``xgb-baseline`` / ``xgb-coral``) —
  re-applies the saved transform.  ``xgb-coral`` requires
  ``--reference-input`` / ``--reference-metadata`` (the train fold's
  merged TSVs) so the held-out cohort is whitened to the *training-only*
  reference distribution; otherwise the alignment would silently leak
  the held-out fingerprint into the reference.

The encoder file layouts written here mirror the train-side artefacts
(``embeddings.tsv`` indexed by sample ID), so the existing
``biomevae-loso-classify`` CLI consumes them without a special case.

Usage::

    biomevae-loso-strict-encode \\
        --model-dir out/loso_strict/crc/diva-beta-vae/FengQ_2015 \\
        --input     out/loso_strict/crc/folds/FengQ_2015/holdout/sgb_table.tsv \\
        --metadata  out/loso_strict/crc/folds/FengQ_2015/holdout/sample_metadata.tsv \\
        --outdir    out/loso_strict/crc/diva-beta-vae/FengQ_2015/holdout
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from biomevae.cli._diva_common import encode_full_dataset
from biomevae.data import load_matrix
from biomevae.loso import _psd_sqrt, load_merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"biomevae-loso-strict-encode: config.json missing from {model_dir}."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _detect_aux_hidden(state: Dict[str, torch.Tensor], default: int = 64) -> int:
    """Read ``aux_hidden`` from the trained DIVA / PhyloDIVA / TAXI state-dict.

    The aux-classifier's first linear layer is the
    ``aux_d.net.0.weight`` tensor of shape ``(aux_hidden, latent_d)``.
    Plain DIVA backbones store it under ``diva.aux_d.*`` (the
    DIVALoss instance is the model's ``self.diva``); PhyloDIVA
    backbones wrap a DIVA inside their own ``self.diva``, so the path
    is doubly nested as ``diva.diva.aux_d.*``.

    The Tree-DTM DIVA family (DIVA / PhyloDIVA / TAXI) does not wrap a
    ``diva`` module — it stores the heads under top-level
    ``domain_classifier`` and ``class_classifier``; both have the same
    aux_hidden width on the first linear, so the first match wins.

    Falls back to ``default`` only if no matching key is found, which
    would mean the state dict is from none of the supported families.
    """
    for key in ("diva.aux_d.net.0.weight", "diva.diva.aux_d.net.0.weight"):
        if key in state:
            return int(state[key].shape[0])
    # Tree-DTM DIVA family stores the heads at top level.
    for key in (
        "domain_classifier.net.0.weight",
        "class_classifier.net.0.weight",
    ):
        if key in state:
            return int(state[key].shape[0])
    # Last-resort glob for any future wrapping depth.
    for k, v in state.items():
        if k.endswith("aux_d.net.0.weight"):
            return int(v.shape[0])
    return int(default)


# ---------------------------------------------------------------------------
# DIVA model rebuild + encode
# ---------------------------------------------------------------------------


def _encode_diva_tree_dtm(
    cfg: Dict, state: Dict[str, torch.Tensor], input_path: Path,
    taxonomy_path: Path, device: torch.device,
    *, phylo: bool = False, taxi: bool = False,
) -> Dict[str, np.ndarray]:
    """Rebuild a DIVATreeDTMVAE / PhyloDIVATreeDTMVAE / TAXIDIVATreeDTMVAE and encode."""
    from biomevae.models.taxonomy_tree import (
        aggregate_leaf_matrix_to_nodes,
        build_taxonomy_graph_from_phyla_tsv,
    )
    from biomevae.models.tree_dtm_vae import build_tree_topology
    from biomevae.taxonomy import load_feature_clades

    model_kwargs = cfg.get("model_kwargs", {}) or {}
    taxg = build_taxonomy_graph_from_phyla_tsv(
        taxonomy_path,
        keep_prefixes=bool(model_kwargs.get("keep_prefixes", False)),
        has_header=bool(model_kwargs.get("taxonomy_has_header", False)),
        on_duplicate_leaf="ignore_same",
    )
    topo = build_tree_topology(taxg)
    leaf_names = [taxg.node_names[nid] for nid in taxg.leaf_ids]

    X_raw, sample_ids = load_matrix(str(input_path), log1p=False)
    feature_clades_input = load_feature_clades(str(input_path))
    name_to_col = {c: i for i, c in enumerate(feature_clades_input)}
    n_leaves = topo.n_leaves
    X_leaf = np.zeros((X_raw.shape[0], n_leaves), dtype=np.float32)
    for li, n in enumerate(leaf_names):
        col = name_to_col.get(n)
        if col is not None:
            X_leaf[:, li] = X_raw[:, col].astype(np.float32)
    X_nodes = aggregate_leaf_matrix_to_nodes(taxg, X_leaf).astype(np.float32)
    x_nodes_t = torch.from_numpy(X_nodes)

    aux_hidden = _detect_aux_hidden(state)
    common_kwargs = dict(
        n_domains=int(cfg["n_domains"]),
        n_classes=int(cfg["n_classes"]),
        topo=topo,
        hidden=int(cfg.get("hidden", 256)),
        latent_d=int(cfg["latent_d"]),
        latent_y=int(cfg["latent_y"]),
        latent_x=int(cfg["latent_x"]),
        encoder_layers=int(cfg.get("encoder_layers", 2)),
        decoder_hidden=int(cfg.get("decoder_hidden", 256)),
        decoder_layers=int(cfg.get("decoder_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        aux_hidden=aux_hidden,
        encoder_pseudocount=float(cfg.get("encoder_pseudocount", 0.5)),
        init_concentration=float(cfg.get("init_concentration", 50.0)),
        likelihood=cfg.get("likelihood", "dirichlet_tree_multinomial"),
    )
    if taxi:
        from biomevae.models.taxi_treedtmvae import TAXIDIVATreeDTMVAE
        # TAXI uses tau/rho naming internally. Drop the y/x kwargs and
        # re-pass them under their TAXI names so the constructor signature
        # is happy.
        taxi_kwargs = dict(common_kwargs)
        latent_tau = int(cfg.get("latent_tau", taxi_kwargs.pop("latent_y")))
        latent_rho = int(cfg.get("latent_rho", taxi_kwargs.pop("latent_x")))
        taxi_kwargs.pop("latent_y", None)
        taxi_kwargs.pop("latent_x", None)
        model = TAXIDIVATreeDTMVAE(
            **taxi_kwargs,
            latent_tau=latent_tau,
            latent_rho=latent_rho,
            critic_hidden=int(cfg.get("critic_hidden", 128)),
            grl_lambda=float(cfg.get("grl_lambda", 1.0)),
            condition_orth_on_domain=bool(
                cfg.get("condition_orth_on_domain", True)
            ),
        ).to(device)
    elif phylo:
        from biomevae.models.phylodiva_treedtmvae import PhyloDIVATreeDTMVAE
        model = PhyloDIVATreeDTMVAE(
            **common_kwargs,
            critic_hidden=int(cfg.get("critic_hidden", 64)),
            critic_condition_on_class=bool(cfg.get("critic_condition_on_class", True)),
            grl_lambda=float(cfg.get("grl_lambda", 1.0)),
        ).to(device)
    else:
        from biomevae.models.diva_treedtmvae import DIVATreeDTMVAE
        model = DIVATreeDTMVAE(**common_kwargs).to(device)

    model.load_state_dict(state)
    model.eval()

    def encode_fn(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = model.encode(x)
        return {"mu_d": enc["mu_d"], "mu_y": enc["mu_y"], "mu_x": enc["mu_x"]}

    embeddings = encode_full_dataset(
        model=model, encode_fn=encode_fn,
        inputs=x_nodes_t, batch_size=128, device=device,
    )
    embeddings["sample_ids"] = np.array(sample_ids, dtype=object)
    return embeddings


def _encode_diva_hyp_philr(
    cfg: Dict, state: Dict[str, torch.Tensor], input_path: Path,
    taxonomy_path: Path, device: torch.device,
    *, phylo: bool = False, taxi: bool = False,
) -> Dict[str, np.ndarray]:
    """Rebuild a DIVA / PhyloDIVA / TAXI HyperbolicPhILRVAE and encode."""
    from biomevae.models.philrvae import build_philrvae_dataset

    data_kind = cfg.get("data_kind", "counts")
    model_kwargs = cfg.get("model_kwargs", {}) or {}
    taxg, X_leaf, _X_nodes, sample_ids, _leaf_names, _ = build_philrvae_dataset(
        Path(input_path), Path(taxonomy_path),
        data_kind=data_kind,
        keep_prefixes=bool(model_kwargs.get("keep_prefixes", False)),
        taxonomy_has_header=bool(model_kwargs.get("taxonomy_has_header", False)),
        allow_missing_leaves=True,
    )

    aux_hidden = _detect_aux_hidden(state)
    common = dict(
        n_domains=int(cfg["n_domains"]),
        n_classes=int(cfg["n_classes"]),
        taxg=taxg,
        curvature=float(cfg.get("curvature", 1.0)),
        hidden=tuple(cfg.get("hidden", [256, 128])),
        latent_d=int(cfg["latent_d"]),
        latent_y=int(cfg["latent_y"]),
        latent_x=int(cfg["latent_x"]),
        dropout=float(cfg.get("dropout", 0.1)),
        aux_hidden=aux_hidden,
        count_pseudocount=float(cfg.get("count_pseudocount", 0.5)),
        relative_pseudocount=float(cfg.get("relative_pseudocount", 1e-6)),
        default_likelihood=cfg.get("likelihood", "philr_gaussian"),
        init_coord_scale=float(cfg.get("init_coord_scale", 0.5)),
        init_concentration=float(cfg.get("init_concentration", 50.0)),
    )
    if taxi:
        from biomevae.models.taxi_hyp_philrvae import TAXIHyperbolicPhILRVAE
        # TAXI uses tau/rho naming internally; drop y/x from the common kwargs
        # and re-pass under the TAXI names so the constructor signature is happy.
        taxi_common = dict(common)
        latent_tau = int(cfg.get("latent_tau", taxi_common.pop("latent_y")))
        latent_rho = int(cfg.get("latent_rho", taxi_common.pop("latent_x")))
        taxi_common.pop("latent_y", None)
        taxi_common.pop("latent_x", None)
        model = TAXIHyperbolicPhILRVAE(
            **taxi_common,
            latent_tau=latent_tau,
            latent_rho=latent_rho,
            critic_hidden=int(cfg.get("critic_hidden", 128)),
            grl_lambda=float(cfg.get("grl_lambda", 1.0)),
            condition_orth_on_domain=bool(
                cfg.get("condition_orth_on_domain", True)
            ),
        ).to(device)
    elif phylo:
        from biomevae.models.phylodiva_hyp_philrvae import PhyloDIVAHyperbolicPhILRVAE
        model = PhyloDIVAHyperbolicPhILRVAE(
            **common,
            critic_hidden=int(cfg.get("critic_hidden", 64)),
            critic_condition_on_class=bool(cfg.get("critic_condition_on_class", True)),
            grl_lambda=float(cfg.get("grl_lambda", 1.0)),
        ).to(device)
    else:
        from biomevae.models.diva_hyp_philrvae import DIVAHyperbolicPhILRVAE
        model = DIVAHyperbolicPhILRVAE(**common).to(device)

    model.load_state_dict(state)
    model.eval()

    def encode_fn(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = model.encode(x, data_kind=data_kind)
        return {"mu_d": enc["mu_d"], "mu_y": enc["mu_y"], "mu_x": enc["mu_x"]}

    embeddings = encode_full_dataset(
        model=model, encode_fn=encode_fn,
        inputs=X_leaf, batch_size=128, device=device,
    )
    embeddings["sample_ids"] = np.array(sample_ids, dtype=object)
    return embeddings


def _encode_phylodiva_hyp_philr(
    cfg: Dict, state: Dict[str, torch.Tensor], input_path: Path,
    taxonomy_path: Path, device: torch.device,
) -> Dict[str, np.ndarray]:
    return _encode_diva_hyp_philr(
        cfg, state, input_path, taxonomy_path, device, phylo=True,
    )


def _encode_phylodiva_betavae(
    cfg: Dict, state: Dict[str, torch.Tensor], input_path: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from biomevae.models.phylodiva_betavae import PhyloDIVABetaVAE

    X_raw, sample_ids = load_matrix(str(input_path), log1p=False)
    if cfg.get("log1p", True):
        x_in = torch.log1p(torch.from_numpy(X_raw.astype(np.float32)))
    else:
        x_in = torch.from_numpy(X_raw.astype(np.float32))

    aux_hidden = _detect_aux_hidden(state)
    model = PhyloDIVABetaVAE(
        input_dim=X_raw.shape[1],
        n_domains=int(cfg["n_domains"]),
        n_classes=int(cfg["n_classes"]),
        hidden=list(cfg.get("hidden", [256, 128, 64])),
        latent_d=int(cfg["latent_d"]),
        latent_y=int(cfg["latent_y"]),
        latent_x=int(cfg["latent_x"]),
        dropout=float(cfg.get("dropout", 0.0)),
        activation=str(cfg.get("activation", "leakyrelu")),
        layer_norm=bool(cfg.get("layer_norm", False)),
        aux_hidden=aux_hidden,
        critic_hidden=int(cfg.get("critic_hidden", 64)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    def encode_fn(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = model.encode(x)
        return {"mu_d": enc["mu_d"], "mu_y": enc["mu_y"], "mu_x": enc["mu_x"]}

    embeddings = encode_full_dataset(
        model=model, encode_fn=encode_fn,
        inputs=x_in, batch_size=128, device=device,
    )
    embeddings["sample_ids"] = np.array(sample_ids, dtype=object)
    return embeddings


def _encode_diva_betavae(
    cfg: Dict, state: Dict[str, torch.Tensor], input_path: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from biomevae.models.diva_betavae import DIVABetaVAE

    X_raw, sample_ids = load_matrix(str(input_path), log1p=False)
    if cfg.get("log1p", True):
        x_in = torch.log1p(torch.from_numpy(X_raw.astype(np.float32)))
    else:
        x_in = torch.from_numpy(X_raw.astype(np.float32))

    aux_hidden = _detect_aux_hidden(state)
    model = DIVABetaVAE(
        input_dim=X_raw.shape[1],
        n_domains=int(cfg["n_domains"]),
        n_classes=int(cfg["n_classes"]),
        hidden=list(cfg.get("hidden", [256, 128, 64])),
        latent_d=int(cfg["latent_d"]),
        latent_y=int(cfg["latent_y"]),
        latent_x=int(cfg["latent_x"]),
        dropout=float(cfg.get("dropout", 0.0)),
        activation=str(cfg.get("activation", "leakyrelu")),
        layer_norm=bool(cfg.get("layer_norm", False)),
        aux_hidden=aux_hidden,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    def encode_fn(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = model.encode(x)
        return {"mu_d": enc["mu_d"], "mu_y": enc["mu_y"], "mu_x": enc["mu_x"]}

    embeddings = encode_full_dataset(
        model=model, encode_fn=encode_fn,
        inputs=x_in, batch_size=128, device=device,
    )
    embeddings["sample_ids"] = np.array(sample_ids, dtype=object)
    return embeddings


def _save_diva_embeddings(
    outdir: Path, embeddings: Dict[str, np.ndarray],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sample_ids = list(embeddings["sample_ids"])
    full = embeddings["mu"]
    pd.DataFrame(
        full, index=sample_ids,
        columns=[f"z{i}" for i in range(full.shape[1])],
    ).to_csv(outdir / "embeddings.tsv", sep="\t")
    for factor in ("z_d", "z_y", "z_x"):
        key = f"mu_{factor.split('_')[1]}"
        arr = embeddings[key]
        pd.DataFrame(
            arr, index=sample_ids,
            columns=[f"{factor}{i}" for i in range(arr.shape[1])],
        ).to_csv(outdir / f"embeddings_{factor}.tsv", sep="\t")


# ---------------------------------------------------------------------------
# XGBoost passthrough featurisers
# ---------------------------------------------------------------------------


def _encode_xgb_baseline(
    cfg: Dict, input_path: Path, outdir: Path,
) -> None:
    """Re-apply the same log1p featurisation used at training time."""
    log1p = bool(cfg.get("log1p", True))
    X, sample_ids = load_matrix(str(input_path), log1p=log1p)
    columns = [f"feat_{i}" for i in range(X.shape[1])]
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=sample_ids, columns=columns).to_csv(
        outdir / "embeddings.tsv", sep="\t",
    )


def _encode_xgb_coral(
    cfg: Dict, input_path: Path, metadata_path: Path,
    reference_input: Path, reference_metadata: Path, outdir: Path,
) -> None:
    """CORAL-align held-out samples to the train-fold reference.

    The training CLI's :func:`biomevae.loso.coral_align` builds the shared
    reference from whichever studies are present in the call.  In strict
    mode we must align held-out samples to a reference derived from the
    *training* studies only — including the held-out cohort in the
    reference computation would re-introduce its fingerprint into the
    target distribution, defeating the strict split.

    We replicate the alignment math inline (the ``_psd_sqrt`` helper
    from :mod:`biomevae.loso` is reused) so the held-out cohort can be
    whitened by its own per-study stats and re-coloured with a
    train-only reference.  Output schema matches the training-time
    ``embeddings.tsv``.
    """
    log1p = bool(cfg.get("log1p", True))
    ridge = float(cfg.get("ridge", 1e-3))
    reference = str(cfg.get("reference", "mean"))
    study_col = str(cfg.get("study_col", "study_name"))

    X_train_raw, train_ids, _, train_meta = load_merged(
        str(reference_input), str(reference_metadata), study_col=study_col,
    )
    X_hold_raw, hold_ids, _, hold_meta = load_merged(
        str(input_path), str(metadata_path), study_col=study_col,
    )

    if log1p:
        X_train = np.log1p(X_train_raw).astype(np.float32)
        X_hold = np.log1p(X_hold_raw).astype(np.float32)
    else:
        X_train = X_train_raw.astype(np.float32)
        X_hold = X_hold_raw.astype(np.float32)

    if X_train.shape[1] != X_hold.shape[1]:
        raise SystemExit(
            "biomevae-loso-strict-encode (xgb-coral): feature count mismatch "
            f"between train ({X_train.shape[1]}) and holdout ({X_hold.shape[1]}) "
            "tables.  loso-strict-fold should have written matched features."
        )

    p = X_train.shape[1]
    dtype = X_train.dtype if np.issubdtype(X_train.dtype, np.floating) else np.float64

    train_studies = train_meta.reindex(train_ids)[study_col].astype(str).fillna("UNKNOWN").to_numpy()
    hold_studies = hold_meta.reindex(hold_ids)[study_col].astype(str).fillna("UNKNOWN").to_numpy()

    def _study_stats(X: np.ndarray, studies: np.ndarray) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for s in sorted({str(v) for v in studies}):
            mask = studies == s
            n_s = int(mask.sum())
            Xs = X[mask].astype(dtype, copy=False)
            if n_s < 2:
                mu_s = Xs.mean(axis=0) if n_s == 1 else np.zeros(p, dtype=dtype)
                cov_s = np.eye(p, dtype=dtype)
            else:
                mu_s = Xs.mean(axis=0)
                cov_s = np.cov(Xs, rowvar=False, ddof=1).astype(dtype, copy=False)
                if cov_s.ndim == 0:
                    cov_s = cov_s.reshape(1, 1)
            out[s] = {"mu": mu_s, "cov": cov_s, "n": n_s}
        return out

    train_stats = _study_stats(X_train, train_studies)
    hold_stats = _study_stats(X_hold, hold_studies)

    if reference == "identity":
        mu_ref = np.zeros(p, dtype=dtype)
        cov_ref = np.eye(p, dtype=dtype)
    elif reference == "largest":
        ref_name = max(train_stats, key=lambda s: train_stats[s]["n"])
        mu_ref = train_stats[ref_name]["mu"].copy()
        cov_ref = train_stats[ref_name]["cov"].copy()
    else:  # "mean" — average of train-fold studies only
        mu_ref = np.mean(
            np.stack([s["mu"] for s in train_stats.values()], axis=0), axis=0,
        )
        cov_ref = np.mean(
            np.stack([s["cov"] for s in train_stats.values()], axis=0), axis=0,
        )

    cov_ref_sqrt = _psd_sqrt(cov_ref, ridge=ridge, invert=False)

    X_aligned = np.empty(X_hold.shape, dtype=dtype)
    for s, stats in hold_stats.items():
        mask = hold_studies == s
        if not mask.any():
            continue
        Xs = X_hold[mask].astype(dtype, copy=False)
        whitener = _psd_sqrt(stats["cov"], ridge=ridge, invert=True)
        X_aligned[mask] = (Xs - stats["mu"]) @ whitener @ cov_ref_sqrt + mu_ref

    columns = [f"feat_{i}" for i in range(X_aligned.shape[1])]
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_aligned, index=hold_ids, columns=columns).to_csv(
        outdir / "embeddings.tsv", sep="\t",
    )


# ---------------------------------------------------------------------------
# CAPDA-VAE passthrough-stacking featuriser
# ---------------------------------------------------------------------------


def _encode_capda_vae(
    cfg: Dict, model_dir: Path, input_path: Path, taxonomy_path: Path,
    outdir: Path, device: str,
) -> None:
    """Apply the trained final CAPDA-VAE to the held-out cohort.

    Emits ``[log1p-species | final-VAE invariant-prob]`` columns — the same
    schema the trainer wrote for the train fold — so the concatenated table
    feeds ``biomevae-loso-classify`` unchanged.  No labels are needed here:
    only the VAE's class-head probabilities are read.
    """
    from biomevae.loso import _read_sgb_table
    from biomevae.models.capda_vae import capda_encode, load_lineage_table

    sgb = _read_sgb_table(input_path)
    feat_clades = sgb["clade_name"].astype(str).tolist()
    sample_cols = sgb.columns[2:].tolist()
    X_raw = (
        sgb[sample_cols].apply(pd.to_numeric, errors="coerce")
        .fillna(0.0).to_numpy(dtype=np.float32).T
    )
    taxonomy = load_lineage_table(
        str(taxonomy_path),
        has_header=bool(cfg.get("taxonomy_has_header", False)),
    )
    state = torch.load(
        model_dir / "model.pt", map_location=torch.device(device),
        weights_only=True,
    )
    emb = capda_encode(
        X_raw, sample_cols, feat_clades, taxonomy, state, cfg, device=device,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    emb.to_csv(outdir / "embeddings.tsv", sep="\t")
    print(
        f"[loso-strict-encode] capda-vae: wrote {emb.shape[0]} × {emb.shape[1]} "
        f"holdout embeddings to {outdir / 'embeddings.tsv'}"
    )


# ---------------------------------------------------------------------------
# Non-DIVA VAEs — delegate to biomevae-embed (already supports them)
# ---------------------------------------------------------------------------


def _encode_via_biomevae_embed(
    model_dir: Path, input_path: Path, taxonomy_path: Path | None,
    outdir: Path, device: str,
) -> None:
    cmd = [
        "biomevae-embed",
        "--input", str(input_path),
        "--model-dir", str(model_dir),
        "--outdir", str(outdir),
        "--device", device,
    ]
    if taxonomy_path is not None:
        cmd.extend(["--taxonomy", str(taxonomy_path)])
    print(f"[loso-strict-encode] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "biomevae-loso-strict-encode",
        description=(
            "Apply a strict-LOSO trained model to a held-out cohort and "
            "write its embeddings (or aligned features for xgb backbones)."
        ),
    )
    ap.add_argument(
        "--model-dir", required=True,
        help=(
            "Directory containing model.pt + config.json from the strict "
            "training step for this fold."
        ),
    )
    ap.add_argument("--input", required=True, help="Held-out sgb_table.tsv.")
    ap.add_argument(
        "--metadata", default=None,
        help="Held-out sample_metadata.tsv (required for xgb-coral).",
    )
    ap.add_argument(
        "--taxonomy", default=None,
        help="phyla.tsv path (required for tree / PhILR / DIVA-tree models).",
    )
    ap.add_argument(
        "--reference-input", default=None,
        help=(
            "Train-fold sgb_table.tsv (xgb-coral only): used to compute "
            "the alignment reference distribution."
        ),
    )
    ap.add_argument(
        "--reference-metadata", default=None,
        help="Train-fold sample_metadata.tsv (xgb-coral only).",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = _read_config(model_dir)
    model_type = str(cfg.get("model_type", "")).lower()
    print(f"[loso-strict-encode] model_type={model_type}")

    if model_type in ("xgb-baseline",):
        _encode_xgb_baseline(cfg, input_path, outdir)
        return

    if model_type in ("xgb-coral",):
        if not args.metadata or not args.reference_input or not args.reference_metadata:
            raise SystemExit(
                "biomevae-loso-strict-encode: xgb-coral requires --metadata, "
                "--reference-input and --reference-metadata."
            )
        _encode_xgb_coral(
            cfg, input_path, Path(args.metadata),
            Path(args.reference_input), Path(args.reference_metadata),
            outdir,
        )
        return

    if model_type in ("capda-vae",):
        if not args.taxonomy:
            raise SystemExit(
                "biomevae-loso-strict-encode: capda-vae requires --taxonomy."
            )
        _encode_capda_vae(
            cfg, model_dir, input_path, Path(args.taxonomy), outdir, args.device,
        )
        return

    if model_type in (
        "tree-dtm-vae",
        "philrvae", "dsvae",
        "hyperbolic-philrvae",
        "euclid", "hyperbolic", "graph_tax", "treeprior",
        "phylo_fusion", "hgvae_zi", "flowxformer",
    ):
        taxonomy_path = Path(args.taxonomy) if args.taxonomy else None
        _encode_via_biomevae_embed(
            model_dir, input_path, taxonomy_path, outdir, args.device,
        )
        return

    if not (
        model_type.startswith("diva-")
        or model_type.startswith("phylodiva-")
        or model_type.startswith("taxi-")
    ):
        raise SystemExit(
            f"biomevae-loso-strict-encode: unsupported model_type "
            f"'{model_type}'."
        )

    # ---- DIVA / PhyloDIVA / TAXI backbones --------------------------------
    state = torch.load(
        model_dir / "model.pt",
        map_location=torch.device(args.device),
        weights_only=True,
    )
    device = torch.device(args.device)

    if model_type == "diva-tree-dtm-vae":
        if not args.taxonomy:
            raise SystemExit("DIVA-Tree-DTM encode requires --taxonomy.")
        embeddings = _encode_diva_tree_dtm(
            cfg, state, input_path, Path(args.taxonomy), device, phylo=False,
        )
    elif model_type == "phylodiva-tree-dtm-vae":
        if not args.taxonomy:
            raise SystemExit("PhyloDIVA-Tree-DTM encode requires --taxonomy.")
        embeddings = _encode_diva_tree_dtm(
            cfg, state, input_path, Path(args.taxonomy), device, phylo=True,
        )
    elif model_type == "taxi-tree-dtm-vae":
        if not args.taxonomy:
            raise SystemExit("TAXI-Tree-DTM encode requires --taxonomy.")
        embeddings = _encode_diva_tree_dtm(
            cfg, state, input_path, Path(args.taxonomy), device, taxi=True,
        )
    elif model_type == "diva-hyp-philrvae":
        if not args.taxonomy:
            raise SystemExit("DIVA-Hyp-PhILR encode requires --taxonomy.")
        embeddings = _encode_diva_hyp_philr(
            cfg, state, input_path, Path(args.taxonomy), device,
        )
    elif model_type == "diva-beta-vae":
        embeddings = _encode_diva_betavae(cfg, state, input_path, device)
    elif model_type == "phylodiva-hyp-philrvae":
        if not args.taxonomy:
            raise SystemExit("PhyloDIVA-Hyp-PhILR encode requires --taxonomy.")
        embeddings = _encode_phylodiva_hyp_philr(
            cfg, state, input_path, Path(args.taxonomy), device,
        )
    elif model_type == "taxi-hyp-philrvae":
        if not args.taxonomy:
            raise SystemExit("TAXI-Hyp-PhILR encode requires --taxonomy.")
        embeddings = _encode_diva_hyp_philr(
            cfg, state, input_path, Path(args.taxonomy), device, taxi=True,
        )
    elif model_type == "phylodiva-beta-vae":
        embeddings = _encode_phylodiva_betavae(cfg, state, input_path, device)
    else:
        raise SystemExit(
            f"biomevae-loso-strict-encode: unhandled DIVA / PhyloDIVA variant "
            f"'{model_type}'."
        )

    _save_diva_embeddings(outdir, embeddings)
    n = embeddings["mu"].shape[0]
    print(
        f"[loso-strict-encode] wrote {n} embeddings + per-factor slices "
        f"to {outdir}/"
    )


if __name__ == "__main__":
    main()
