"""End-to-end smoke tests for TAXI-Tree-DTM-VAE.

Covers:

* CLI smoke-train on a tiny synthetic merged dataset (2 studies, 2 classes,
  6 leaves) — checks that the canonical artefacts are emitted and that the
  TAXI-specific extra-loss columns appear in the training log.
* Strict-LOSO encode round-trip: re-loads ``model.pt`` + ``config.json``
  from the smoke run via :func:`biomevae.cli.loso_strict_encode._encode_diva_tree_dtm`
  and asserts the encoder produces sensible embeddings.
* Unit checks on the conditional invariance machinery: the GRL truly
  gates the encoder gradient (λ=0 ⇒ zero gradient, λ=1 ⇒ nonzero), and
  ``z_tau`` does NOT receive adversarial gradient at any setting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Synthetic dataset (same fixture shape as test_phylodiva_smoke)
# ---------------------------------------------------------------------------


def _write_synthetic_dataset(tmp_path: Path, n_per_study: int = 10) -> dict:
    leaves = ["sgb1", "sgb2", "sgb3", "sgb4", "sgb5", "sgb6"]
    studies = ["StudyA", "StudyB"]
    classes = ["healthy", "CRC"]
    rng = np.random.RandomState(0)

    sample_ids: list[str] = []
    rows = []
    meta = []
    for s in studies:
        for c in classes:
            for k in range(n_per_study // 2):
                sid = f"{s}_{c}_{k:02d}"
                sample_ids.append(sid)
                base = rng.poisson(lam=20.0, size=6).astype(float)
                if c == "CRC":
                    base[0] *= 3.0
                    base[1] *= 3.0
                if s == "StudyB":
                    base[3] *= 2.5
                    base[4] *= 2.5
                rows.append(base)
                meta.append({"sample_id": sid, "study_name": s, "disease": c})

    X = np.array(rows)

    sgb_df = pd.DataFrame(X.T, index=leaves, columns=sample_ids)
    sgb_df.insert(0, "NCBI_tax_id", "0")
    sgb_df.index.name = "clade_name"
    sgb_path = tmp_path / "sgb_table.tsv"
    sgb_df.to_csv(sgb_path, sep="\t")

    phyla = [
        ["sgb1", "k__Bacteria", "p__P1", "c__C1", "f__F1", "g__G1"],
        ["sgb2", "k__Bacteria", "p__P1", "c__C1", "f__F1", "g__G1"],
        ["sgb3", "k__Bacteria", "p__P1", "c__C2", "f__F2", "g__G2"],
        ["sgb4", "k__Bacteria", "p__P2", "c__C3", "f__F3", "g__G3"],
        ["sgb5", "k__Bacteria", "p__P2", "c__C3", "f__F4", "g__G4"],
        ["sgb6", "k__Bacteria", "p__P2", "c__C4", "f__F5", "g__G5"],
    ]
    phyla_path = tmp_path / "phyla.tsv"
    pd.DataFrame(phyla).to_csv(phyla_path, sep="\t", index=False, header=False)

    meta_df = pd.DataFrame(meta)
    meta_path = tmp_path / "sample_metadata.tsv"
    meta_df.to_csv(meta_path, sep="\t", index=False)

    return {
        "sgb": str(sgb_path),
        "phyla": str(phyla_path),
        "metadata": str(meta_path),
    }


def _taxi_argv(paths: dict, outdir: Path) -> list[str]:
    return [
        "--input", paths["sgb"],
        "--taxonomy", paths["phyla"],
        "--metadata", paths["metadata"],
        "--label-col", "disease",
        "--study-col", "study_name",
        "--outdir", str(outdir),
        "--epochs", "2",
        "--batch-size", "8",
        "--early-stop", "0",
        "--device", "cpu",
        "--seed", "0",
        "--hidden", "16",
        "--decoder-hidden", "16",
        "--latent-d", "2",
        "--latent-tau", "4",
        "--latent-rho", "4",
        "--encoder-layers", "1",
        "--decoder-layers", "1",
        "--data-kind", "counts",
        "--likelihood", "dirichlet_tree_multinomial",
        "--critic-hidden", "8",
        "--lambda-cond-critic", "0.1",
        "--lambda-cond-coral", "0.1",
        "--lambda-tree-smooth", "0.01",
        "--lambda-orth", "0.05",
        "--lambda-tau-aux", "0.1",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_smoke_taxi_tree_dtm_vae(tmp_path):
    from biomevae.cli.vae_train_taxi_tree_dtm_vae import main

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "taxi-tree-dtm-vae"
    main(_taxi_argv(paths, outdir))

    assert (outdir / "model.pt").exists()
    assert (outdir / "embeddings.tsv").exists()
    assert (outdir / "embeddings_z_y.tsv").exists()
    assert (outdir / "embeddings_z_x.tsv").exists()
    assert (outdir / "embeddings_z_d.tsv").exists()
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == "taxi-tree-dtm-vae"
    assert cfg["latent_tau"] == 4
    assert cfg["latent_rho"] == 4

    log = pd.read_csv(outdir / "training_log.tsv", sep="\t")
    # Extras must appear with non-NaN values once λ>0 is used.
    for col in (
        "train_cond_critic",
        "train_cond_coral",
        "train_tree_smooth",
        "train_orth",
        "train_tau_aux",
    ):
        assert col in log.columns, f"missing {col}"
        assert np.isfinite(log[col].iloc[-1])


def test_strict_encode_round_trip_taxi(tmp_path):
    """Re-instantiate TAXI from saved config/state and re-encode the input."""
    from biomevae.cli.vae_train_taxi_tree_dtm_vae import main as train_main
    from biomevae.cli.loso_strict_encode import _encode_diva_tree_dtm

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "taxi"
    train_main(_taxi_argv(paths, outdir))

    cfg = json.loads((outdir / "config.json").read_text())
    state = torch.load(
        outdir / "model.pt", map_location="cpu", weights_only=True,
    )

    emb = _encode_diva_tree_dtm(
        cfg, state,
        Path(paths["sgb"]), Path(paths["phyla"]),
        torch.device("cpu"), taxi=True,
    )
    # Shape sanity: mu_y == latent_tau, mu_x == latent_rho.
    assert emb["mu_y"].shape[1] == cfg["latent_tau"]
    assert emb["mu_x"].shape[1] == cfg["latent_rho"]
    assert emb["mu_d"].shape[1] == cfg["latent_d"]
    assert emb["mu"].shape[1] == (
        cfg["latent_d"] + cfg["latent_tau"] + cfg["latent_rho"]
    )
    n_samples = sum(
        1 for line in open(paths["sgb"], "r")
        if line.strip()
    ) - 1  # minus header row in sgb table
    # sgb file has one row per leaf; sample count comes from columns
    sgb_header = open(paths["sgb"]).readline().rstrip("\n").split("\t")
    n_samples = len(sgb_header) - 2  # drop clade_name + NCBI_tax_id
    assert emb["mu"].shape[0] == n_samples


def test_taxi_grl_gates_residual_only(tmp_path):
    """The TAXI conditional critic must:

    * pass non-zero adversarial gradient to ``z_rho`` encoder when ``λ>0``,
    * pass exactly zero gradient to ``z_rho`` encoder when ``λ=0``,
    * NEVER pass adversarial gradient to ``z_tau`` encoder (because the
      critic stops gradient on z_tau and y_context).
    """
    from biomevae.cli.vae_train_diva_tree_dtm_vae import _build_dataset
    from biomevae.models.taxi_treedtmvae import TAXIDIVATreeDTMVAE
    import argparse

    paths = _write_synthetic_dataset(tmp_path)
    args = argparse.Namespace(
        input=paths["sgb"], metadata=paths["metadata"], taxonomy=paths["phyla"],
        study_col="study_name", label_col="disease",
        data_kind="counts", likelihood="dirichlet_tree_multinomial",
        keep_prefixes=False, taxonomy_has_header=False,
    )
    ds, X_nodes, _leaves, topo = _build_dataset(args)

    model = TAXIDIVATreeDTMVAE(
        n_domains=len(ds.domain_classes),
        n_classes=len(ds.class_classes),
        topo=topo,
        hidden=16, decoder_hidden=16,
        latent_d=2, latent_tau=4, latent_rho=4,
        encoder_layers=1, decoder_layers=1,
        critic_hidden=8, dropout=0.0,
        likelihood="dirichlet_tree_multinomial",
    )
    model.train()

    rho_lin = next(m for m in model.enc_x.modules() if isinstance(m, torch.nn.Linear))
    tau_lin = next(m for m in model.enc_y.modules() if isinstance(m, torch.nn.Linear))

    # ── λ=1: residual encoder must move; tau encoder must not. ──
    model.zero_grad(set_to_none=True)
    model.set_grl_lambda(1.0)
    out = model(X_nodes, domain=ds.domain, klass=ds.klass)
    extra, _ = model.extra_losses(
        out, ds.domain, ds.klass,
        lambda_cond_critic=1.0,
        lambda_cond_coral=0.0,
        lambda_tree_smooth=0.0,
        lambda_orth=0.0,
        lambda_tau_aux=0.0,
    )
    extra.backward()
    assert rho_lin.weight.grad is not None
    assert rho_lin.weight.grad.abs().mean().item() > 0.0, (
        "λ=1: residual encoder z_rho received no adversarial gradient."
    )
    # tau encoder must be insulated: only adversarial loss is active and
    # the critic stop-grads z_tau; so the gradient is exactly zero.
    if tau_lin.weight.grad is None:
        tau_grad = 0.0
    else:
        tau_grad = tau_lin.weight.grad.abs().max().item()
    assert tau_grad < 1e-9, (
        f"λ=1: tau encoder z_tau received adversarial gradient {tau_grad} "
        "but should be insulated (critic stop-grads z_tau)."
    )

    # ── λ=0: residual encoder must NOT move from the critic. ──
    model.zero_grad(set_to_none=True)
    model.set_grl_lambda(0.0)
    out2 = model(X_nodes, domain=ds.domain, klass=ds.klass)
    extra2, _ = model.extra_losses(
        out2, ds.domain, ds.klass,
        lambda_cond_critic=1.0,
        lambda_cond_coral=0.0,
        lambda_tree_smooth=0.0,
        lambda_orth=0.0,
        lambda_tau_aux=0.0,
    )
    extra2.backward()
    if rho_lin.weight.grad is None:
        rho_grad0 = 0.0
    else:
        rho_grad0 = rho_lin.weight.grad.abs().max().item()
    assert rho_grad0 < 1e-9, (
        f"λ=0: residual encoder received gradient {rho_grad0} but the GRL "
        "should null it."
    )


def test_make_class_context_uses_labels_when_present():
    from biomevae.models.taxi_treedtmvae import make_class_context

    logits = torch.tensor([[0.1, 5.0], [10.0, 0.1], [0.0, 0.0]])
    klass = torch.tensor([0, 1, -1])  # third is unlabeled
    ctx = make_class_context(logits, klass, n_classes=2)

    # Labeled rows must be exact one-hot.
    assert torch.allclose(ctx[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(ctx[1], torch.tensor([0.0, 1.0]))
    # Unlabeled row uses softmax of logits; for [0, 0] -> [0.5, 0.5].
    assert torch.allclose(ctx[2], torch.tensor([0.5, 0.5]), atol=1e-6)
    # Detached: no grad fn.
    assert not ctx.requires_grad


def test_conditional_coral_falls_back_to_zero_when_underdetermined():
    """When no class/domain stratum has enough weighted mass, the loss must
    return exactly 0 with a real tensor on the graph (not NaN or scalar)."""
    from biomevae.models.taxi_treedtmvae import conditional_coral_by_class_domain

    z = torch.randn(4, 3, requires_grad=True)
    domain = torch.tensor([0, 0, 0, 0])  # only one domain
    y_context = torch.eye(2)[torch.tensor([0, 0, 1, 1])]
    loss = conditional_coral_by_class_domain(
        z, domain, y_context, n_domains=2, min_weight=2.0,
    )
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_smoke_taxi_hyp_philrvae(tmp_path):
    """CLI smoke-train of the TAXI hyperbolic-PhILR variant."""
    pytest.importorskip("geoopt")
    from biomevae.cli.vae_train_taxi_hyp_philrvae import main

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "taxi-hyp-philrvae"
    argv = [
        "--input", paths["sgb"],
        "--taxonomy", paths["phyla"],
        "--metadata", paths["metadata"],
        "--label-col", "disease", "--study-col", "study_name",
        "--outdir", str(outdir),
        "--epochs", "2", "--batch-size", "8", "--early-stop", "0",
        "--device", "cpu", "--seed", "0",
        "--hidden", "16",
        "--latent-d", "2", "--latent-tau", "4", "--latent-rho", "4",
        "--data-kind", "relative", "--likelihood", "philr_gaussian",
        "--critic-hidden", "8",
        "--lambda-cond-critic", "0.1",
        "--lambda-cond-coral", "0.1",
        "--lambda-philr-smooth", "0.01",
        "--lambda-orth", "0.05",
        "--lambda-tau-aux", "0.1",
    ]
    main(argv)

    assert (outdir / "model.pt").exists()
    assert (outdir / "embeddings.tsv").exists()
    assert (outdir / "embeddings_z_y.tsv").exists()
    assert (outdir / "embeddings_z_x.tsv").exists()
    assert (outdir / "embeddings_z_d.tsv").exists()
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == "taxi-hyp-philrvae"
    assert cfg["latent_tau"] == 4
    assert cfg["latent_rho"] == 4

    log = pd.read_csv(outdir / "training_log.tsv", sep="\t")
    for col in (
        "train_cond_critic",
        "train_cond_coral",
        "train_philr_smooth",
        "train_orth",
        "train_tau_aux",
    ):
        assert col in log.columns, f"missing {col}"
        assert np.isfinite(log[col].iloc[-1])


def test_strict_encode_round_trip_taxi_hyp_philrvae(tmp_path):
    """Re-instantiate TAXI hyp-philr from saved config/state and re-encode."""
    pytest.importorskip("geoopt")
    from biomevae.cli.vae_train_taxi_hyp_philrvae import main as train_main
    from biomevae.cli.loso_strict_encode import _encode_diva_hyp_philr

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "taxi-hyp-philrvae"
    argv = [
        "--input", paths["sgb"],
        "--taxonomy", paths["phyla"],
        "--metadata", paths["metadata"],
        "--label-col", "disease", "--study-col", "study_name",
        "--outdir", str(outdir),
        "--epochs", "2", "--batch-size", "8", "--early-stop", "0",
        "--device", "cpu", "--seed", "0",
        "--hidden", "16",
        "--latent-d", "2", "--latent-tau", "4", "--latent-rho", "4",
        "--data-kind", "relative", "--likelihood", "philr_gaussian",
        "--critic-hidden", "8",
        "--lambda-cond-critic", "0.1",
    ]
    train_main(argv)

    cfg = json.loads((outdir / "config.json").read_text())
    state = torch.load(
        outdir / "model.pt", map_location="cpu", weights_only=True,
    )
    emb = _encode_diva_hyp_philr(
        cfg, state, Path(paths["sgb"]), Path(paths["phyla"]),
        torch.device("cpu"), taxi=True,
    )
    assert emb["mu_y"].shape[1] == cfg["latent_tau"]
    assert emb["mu_x"].shape[1] == cfg["latent_rho"]
    assert emb["mu_d"].shape[1] == cfg["latent_d"]
    assert emb["mu"].shape[1] == (
        cfg["latent_d"] + cfg["latent_tau"] + cfg["latent_rho"]
    )


def test_tree_contrast_smoothness_shift_invariant(tmp_path):
    """Adding the same constant to every sibling group's logits must not
    change the smoothness penalty — that is the gauge invariance the
    centering step is meant to enforce."""
    from biomevae.cli.vae_train_diva_tree_dtm_vae import _build_dataset
    from biomevae.models.taxi_treedtmvae import (
        TAXIDIVATreeDTMVAE, tree_contrast_smoothness,
    )
    import argparse

    paths = _write_synthetic_dataset(tmp_path)
    args = argparse.Namespace(
        input=paths["sgb"], metadata=paths["metadata"], taxonomy=paths["phyla"],
        study_col="study_name", label_col="disease",
        data_kind="counts", likelihood="dirichlet_tree_multinomial",
        keep_prefixes=False, taxonomy_has_header=False,
    )
    ds, X_nodes, _leaves, topo = _build_dataset(args)

    model = TAXIDIVATreeDTMVAE(
        n_domains=len(ds.domain_classes),
        n_classes=len(ds.class_classes),
        topo=topo,
        hidden=16, decoder_hidden=16,
        latent_d=2, latent_tau=4, latent_rho=4,
        encoder_layers=1, decoder_layers=1,
        critic_hidden=8, dropout=0.0,
        likelihood="dirichlet_tree_multinomial",
    )
    model.eval()
    with torch.no_grad():
        out = model(X_nodes)
        edge_logits = out["edge_logits"]

    base = tree_contrast_smoothness(
        edge_logits,
        model.taxi_edge_to_group,
        model.taxi_group_sizes,
        model.taxi_parent_child_edge_pairs,
        model.taxi_edge_length,
    )

    # Add a per-sibling-group constant — the softmax gauge.  After
    # centering, the smoothness penalty must be invariant.
    group_shifts = torch.randn(model.taxi_group_sizes.numel())
    per_edge_shift = group_shifts[model.taxi_edge_to_group]
    shifted = edge_logits + per_edge_shift.unsqueeze(0)

    shifted_loss = tree_contrast_smoothness(
        shifted,
        model.taxi_edge_to_group,
        model.taxi_group_sizes,
        model.taxi_parent_child_edge_pairs,
        model.taxi_edge_length,
    )
    assert torch.allclose(base, shifted_loss, atol=1e-5), (
        f"smoothness not shift-invariant: {float(base)} vs {float(shifted_loss)}"
    )
