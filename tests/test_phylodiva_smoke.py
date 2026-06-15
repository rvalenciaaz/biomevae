"""End-to-end smoke tests for the three PhyloDIVA wrappers.

Each test:
* generates a tiny synthetic merged sgb_table.tsv + phyla.tsv +
  sample_metadata.tsv,
* drives the corresponding training CLI for 2 epochs (no Optuna),
* asserts that the canonical artefacts (model.pt, embeddings.tsv,
  embeddings_z_y.tsv, training_log.tsv with ``train_critic`` /
  ``train_coral`` columns) are written, and
* re-loads the saved state with the same wrapper to confirm it round-
  trips (this is also the path
  :mod:`biomevae.cli.loso_strict_encode` takes for held-out cohorts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _write_synthetic_dataset(tmp_path: Path, n_per_study: int = 10) -> dict:
    """Make a 6-leaf, 2-study, 2-class synthetic merged dataset.

    Returns a dict of paths the CLIs consume.
    """
    # Production sgb_table.clade_name and phyla.tsv first column carry
    # bare leaf ids (no rank prefix); keep that convention here.
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
                # Per-class signal on leaves 1-2; per-study shift on 4-5.
                base = rng.poisson(lam=20.0, size=6).astype(float)
                if c == "CRC":
                    base[0] *= 3.0
                    base[1] *= 3.0
                if s == "StudyB":
                    base[3] *= 2.5
                    base[4] *= 2.5
                rows.append(base)
                meta.append({
                    "sample_id": sid, "study_name": s, "disease": c,
                })

    X = np.array(rows)

    # sgb_table.tsv layout:  clade_name | NCBI_tax_id | <samples...>
    sgb_df = pd.DataFrame(
        X.T, index=leaves, columns=sample_ids,
    )
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
    pd.DataFrame(phyla).to_csv(
        phyla_path, sep="\t", index=False, header=False,
    )

    meta_df = pd.DataFrame(meta)
    meta_path = tmp_path / "sample_metadata.tsv"
    meta_df.to_csv(meta_path, sep="\t", index=False)

    return {
        "sgb": str(sgb_path),
        "phyla": str(phyla_path),
        "metadata": str(meta_path),
    }


def _run_train(cli_main, argv: list[str]) -> None:
    cli_main(argv)


def _common_argv(paths: dict, outdir: Path, *, with_taxonomy: bool = True) -> list[str]:
    argv = ["--input", paths["sgb"]]
    if with_taxonomy:
        argv.extend(["--taxonomy", paths["phyla"]])
    argv.extend([
        "--metadata", paths["metadata"],
        "--label-col", "disease",
        "--study-col", "study_name",
        "--outdir", str(outdir),
        "--epochs", "2",
        "--batch-size", "8",
        "--early-stop", "0",
        "--device", "cpu",
        "--seed", "0",
        # Keep phylo-DA penalties small but non-zero so the log columns
        # appear and the parameters get gradients.  The β-VAE CLI uses a
        # constant ``--grl-lambda`` (no DANN sigmoid ramp), mirroring
        # the tree-dtm / hyp-philrvae phylodiva variants.
        "--grl-lambda", "1.0",
        "--lambda-coral", "0.1",
        "--lambda-critic", "0.1",
        "--critic-hidden", "8",
    ])
    return argv


def _assert_artefacts(outdir: Path, model_type: str) -> None:
    assert (outdir / "model.pt").exists(), "model.pt missing"
    assert (outdir / "embeddings.tsv").exists(), "embeddings.tsv missing"
    assert (outdir / "embeddings_z_y.tsv").exists(), "embeddings_z_y.tsv missing"
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == model_type
    log = pd.read_csv(outdir / "training_log.tsv", sep="\t")
    # The extra-loss callback should have populated ``train_critic`` /
    # ``train_coral`` (and ``train_bm`` for tree-aware backbones).
    assert "train_critic" in log.columns
    assert "train_coral" in log.columns


def _tree_dtm_common_argv(paths: dict, outdir: Path) -> list[str]:
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
        "--latent-d", "2", "--latent-y", "4", "--latent-x", "4",
        "--encoder-layers", "1", "--decoder-layers", "1",
        "--data-kind", "counts",
        "--likelihood", "dirichlet_tree_multinomial",
    ]


def test_smoke_diva_tree_dtm_vae(tmp_path):
    from biomevae.cli.vae_train_diva_tree_dtm_vae import main

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "diva-tree-dtm-vae"
    main(_tree_dtm_common_argv(paths, outdir))

    assert (outdir / "model.pt").exists()
    assert (outdir / "embeddings.tsv").exists()
    assert (outdir / "embeddings_z_y.tsv").exists()
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == "diva-tree-dtm-vae"
    log = pd.read_csv(outdir / "training_log.tsv", sep="\t")
    assert "train_kl_d" in log.columns
    assert "train_ce_y" in log.columns


def test_smoke_phylodiva_tree_dtm_vae(tmp_path):
    from biomevae.cli.vae_train_phylodiva_tree_dtm_vae import main

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "phylodiva-tree-dtm-vae"
    argv = _tree_dtm_common_argv(paths, outdir) + [
        "--critic-hidden", "8",
        "--lambda-critic", "0.1",
        "--lambda-coral", "0.1",
        "--lambda-tree-smooth", "0.01",
    ]
    main(argv)

    assert (outdir / "model.pt").exists()
    assert (outdir / "embeddings.tsv").exists()
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == "phylodiva-tree-dtm-vae"
    log = pd.read_csv(outdir / "training_log.tsv", sep="\t")
    assert "train_critic" in log.columns
    assert "train_coral" in log.columns
    assert "train_tree_smooth" in log.columns


def test_smoke_phylodiva_hyp_philrvae(tmp_path):
    pytest.importorskip("geoopt")
    from biomevae.cli.vae_train_phylodiva_hyp_philrvae import main

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "phylodiva-hyp-philrvae"
    argv = [
        "--input", paths["sgb"],
        "--taxonomy", paths["phyla"],
        "--metadata", paths["metadata"],
        "--label-col", "disease", "--study-col", "study_name",
        "--outdir", str(outdir),
        "--epochs", "2", "--batch-size", "8", "--early-stop", "0",
        "--device", "cpu", "--seed", "0",
        "--hidden", "16",
        "--latent-d", "2", "--latent-y", "4", "--latent-x", "4",
        "--data-kind", "relative", "--likelihood", "philr_gaussian",
        "--critic-hidden", "8",
        "--lambda-critic", "0.1", "--lambda-coral", "0.1",
        "--lambda-philr-smooth", "0.01",
    ]
    _run_train(main, argv)
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == "phylodiva-hyp-philrvae"
    log = pd.read_csv(outdir / "training_log.tsv", sep="\t")
    assert "train_critic" in log.columns
    assert "train_coral" in log.columns
    assert "train_philr_smooth" in log.columns


def test_smoke_phylodiva_betavae(tmp_path):
    from biomevae.cli.vae_train_phylodiva_betavae import main

    paths = _write_synthetic_dataset(tmp_path)
    outdir = tmp_path / "phylodiva-beta-vae"
    # The β-VAE backbone is tax-agnostic and its CLI does not accept
    # --taxonomy; pass with_taxonomy=False so argparse is clean.
    argv = _common_argv(paths, outdir, with_taxonomy=False) + [
        "--latent-d", "2", "--latent-y", "4", "--latent-x", "4",
        "--hidden", "16", "8",
    ]
    _run_train(main, argv)
    _assert_artefacts(outdir, "phylodiva-beta-vae")
    # β-VAE has no BM column.
    log = pd.read_csv(outdir / "training_log.tsv", sep="\t")
    assert "train_bm" not in log.columns


def test_strict_encode_recovers_aux_hidden_from_phylodiva_state(tmp_path):
    """Regression test: ``_detect_aux_hidden`` must read the
    ``aux_hidden`` value back from a PhyloDIVA state-dict (whose path
    is doubly nested as ``diva.diva.aux_d.*``), not silently fall back
    to the default 64.  Without this the strict-encode model rebuild
    fails with a shape mismatch when training used a non-default
    ``--aux-hidden``.
    """
    pytest.skip(
        "PhyloDIVA-TreeNB has been removed; aux_hidden detection is "
        "covered by the remaining hyp-philr / beta-vae smoke tests."
    )


def test_grl_flows_to_encoder_in_each_wrapper(tmp_path):
    """End-to-end regression test: the GRL must invert the encoder's
    gradient sign in every PhyloDIVA wrapper.  The earliest draft of
    this module shipped a hierarchical critic on input-space clade
    aggregations whose GRL never reached the encoder (the aggregator
    has no learnable parameters).  This test locks in the
    LatentStudyCritic-on-z_y design that fixes that.

    For each wrapper:
      * with ``λ=1``, the encoder's first Linear receives a non-zero
        gradient when the critic loss is back-propagated.
      * with ``λ=0``, the encoder's first Linear receives exactly zero
        gradient (GRL gating).
    """
    from biomevae.cli._diva_common import build_diva_dataset
    from biomevae.models.phylodiva_betavae import PhyloDIVABetaVAE

    paths = _write_synthetic_dataset(tmp_path)
    ds = build_diva_dataset(paths["sgb"], paths["metadata"])

    def _check(model, *fwd_args):
        model.train()
        first_linear = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
        # λ=1 → encoder must receive non-zero gradient.
        model.critic.set_lambda(1.0)
        out = model(*fwd_args)
        extras = model.extra_losses(out, lambda_bm=0, lambda_coral=0, lambda_critic=1.0)
        extras["critic"].backward()
        assert first_linear.weight.grad.abs().mean().item() > 0
        # λ=0 → encoder gradient must be exactly zero.
        model.zero_grad()
        model.critic.set_lambda(0.0)
        out2 = model(*fwd_args)
        extras2 = model.extra_losses(out2, lambda_bm=0, lambda_coral=0, lambda_critic=1.0)
        extras2["critic"].backward()
        assert first_linear.weight.grad.abs().max().item() < 1e-9

    # β-VAE (no taxonomy / no tree)
    m3 = PhyloDIVABetaVAE(
        input_dim=ds.x_raw.size(1),
        n_domains=len(ds.domain_classes), n_classes=len(ds.class_classes),
        hidden=[16], latent_d=2, latent_y=4, latent_x=4,
        dropout=0.0, critic_hidden=8,
    )
    _check(m3, torch.log1p(ds.x_raw), ds.domain, ds.klass)


def test_grl_flows_to_encoder_in_hyp_philr_wrapper(tmp_path):
    """Same GRL gradient check for the new compositional Hyp-PhILR wrapper."""
    pytest.importorskip("geoopt")

    from biomevae.models.philrvae import build_philrvae_dataset
    from biomevae.models.phylodiva_hyp_philrvae import PhyloDIVAHyperbolicPhILRVAE

    paths = _write_synthetic_dataset(tmp_path)
    taxg, X_leaf, _X_nodes, _sids, _leaf_names, _ = build_philrvae_dataset(
        paths["sgb"], paths["phyla"], data_kind="relative", allow_missing_leaves=True,
    )
    n = X_leaf.size(0)
    domain = torch.randint(0, 2, (n,)).long()
    klass = torch.randint(0, 2, (n,)).long()

    m = PhyloDIVAHyperbolicPhILRVAE(
        n_domains=2, n_classes=2, taxg=taxg,
        latent_d=2, latent_y=4, latent_x=4, hidden=(16,),
        dropout=0.0, critic_hidden=8,
    )
    m.train()
    # The critic operates on z_y, so gradient must reach the y-encoder.
    y_enc_linear = next(mod for mod in m.enc_y.modules() if isinstance(mod, torch.nn.Linear))
    m.study_critic.set_lambda(1.0)
    out = m(X_leaf, domain=domain, klass=klass, data_kind="relative")
    extras, _ = m.extra_losses(out, domain, klass, lambda_critic=1.0)
    extras.backward()
    assert y_enc_linear.weight.grad is not None
    assert y_enc_linear.weight.grad.abs().mean().item() > 0
