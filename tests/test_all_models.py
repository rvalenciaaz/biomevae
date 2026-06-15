"""
Comprehensive end-to-end tests for every biomevae CLI script.

Generates a small synthetic dataset (20 taxa, 30 samples, 3 taxonomy levels)
and runs train -> embed -> test for all 10 model variants, plus NMF baseline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import importlib

import numpy as np
import pandas as pd
import pytest

_has_geoopt = importlib.util.find_spec("geoopt") is not None
_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None

# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------
N_TAXA = 20
N_SAMPLES = 30
SEED = 42

# Small network: fast training
FAST_TRAIN_ARGS = [
    "--epochs", "5",
    "--batch-size", "8",
    "--hidden", "32", "16",
    "--latent-dim", "4",
    "--lr", "1e-3",
    "--early-stop", "0",
    "--device", "cpu",
    "--log1p",
    "--seed", "42",
]


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> dict:
    """Create tiny synthetic abundance table + taxonomy file."""
    rng = np.random.RandomState(SEED)
    base_dir = tmp_path_factory.mktemp("synth")
    sgb_path = base_dir / "sgb_table.tsv"
    tax_path = base_dir / "phyla.tsv"

    # Build abundance table: rows=taxa, cols=clade_name + NCBI_tax_id + samples
    clade_names = [f"t__SGB{i:03d}" for i in range(N_TAXA)]
    tax_ids = [f"{1000+i}" for i in range(N_TAXA)]
    sample_names = [f"sample_{i:03d}" for i in range(N_SAMPLES)]
    # Sparse counts with a few nonzero entries
    counts = rng.poisson(lam=2.0, size=(N_TAXA, N_SAMPLES)).astype(int)
    # Make it somewhat sparse
    mask = rng.random(size=(N_TAXA, N_SAMPLES)) < 0.4
    counts[mask] = 0

    rows = []
    for i in range(N_TAXA):
        row = [clade_names[i], tax_ids[i]] + [str(c) for c in counts[i]]
        rows.append("\t".join(row))
    header = "\t".join(["clade_name", "NCBI_tax_id"] + sample_names)
    sgb_path.write_text(header + "\n" + "\n".join(rows) + "\n")

    # Build taxonomy file: clade_name + k + p + c + o + f + g + s
    kingdoms = ["k__Bacteria", "k__Archaea"]
    phyla = ["p__Firmicutes", "p__Bacteroidetes", "p__Proteobacteria"]
    classes = ["c__ClassA", "c__ClassB"]
    orders = ["o__OrderA", "o__OrderB"]
    families = ["f__FamilyA", "f__FamilyB", "f__FamilyC"]
    genera = [f"g__Genus{i}" for i in range(5)]

    tax_rows = []
    for i in range(N_TAXA):
        k = kingdoms[i % len(kingdoms)]
        p = phyla[i % len(phyla)]
        c = classes[i % len(classes)]
        o = orders[i % len(orders)]
        f = families[i % len(families)]
        g = genera[i % len(genera)]
        s = f"s__Species{i}"
        tax_rows.append("\t".join([clade_names[i], k, p, c, o, f, g, s]))
    tax_path.write_text("\n".join(tax_rows) + "\n")

    return {
        "sgb": str(sgb_path),
        "tax": str(tax_path),
        "base_dir": str(base_dir),
    }


def _run(cmd: list[str], label: str, timeout: int = 300):
    """Run a CLI command and assert it succeeds."""
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
    assert result.returncode == 0, f"{label} failed with rc={result.returncode}\n{result.stderr[-2000:]}"
    return result


def _check_training_outputs(outdir: str, model_type: str = "standard"):
    """Verify expected output files from a training run."""
    assert os.path.isfile(os.path.join(outdir, "config.json")), "config.json missing"
    assert os.path.isfile(os.path.join(outdir, "model.pt")), "model.pt missing"
    assert os.path.isfile(os.path.join(outdir, "training_log.tsv")), "training_log.tsv missing"
    cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
    assert "latent_dim" in cfg
    log = pd.read_csv(os.path.join(outdir, "training_log.tsv"), sep="\t")
    assert len(log) > 0, "training log is empty"
    assert "epoch" in log.columns


def _check_embed_outputs(outdir: str):
    """Verify expected output files from embed."""
    emb_path = os.path.join(outdir, "embeddings.tsv")
    assert os.path.isfile(emb_path), "embeddings.tsv missing"
    df = pd.read_csv(emb_path, sep="\t", index_col=0)
    assert df.shape[0] == N_SAMPLES, f"Expected {N_SAMPLES} rows, got {df.shape[0]}"
    assert df.shape[1] > 0, "embeddings have no columns"


def _check_test_outputs(outdir: str):
    """Verify expected output files from test."""
    rpt_path = os.path.join(outdir, "test_report.json")
    assert os.path.isfile(rpt_path), "test_report.json missing"
    report = json.loads(Path(rpt_path).read_text())
    assert "reconstruction" in report
    assert "kl_mean" in report


# ===========================================================================
# TRAINING TESTS
# ===========================================================================

class TestBaseVAE:
    """biomevae-train (beta-VAE, euclid)"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "base_vae")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", outdir] + FAST_TRAIN_ARGS,
            "biomevae-train (base beta-VAE)",
        )
        _check_training_outputs(outdir)

    def test_embed(self, synthetic_data, tmp_path):
        # First train
        model_dir = str(tmp_path / "base_vae_model")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", model_dir] + FAST_TRAIN_ARGS,
            "train for embed test",
        )
        # Then embed
        emb_dir = str(tmp_path / "base_vae_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", emb_dir,
             "--device", "cpu", "--export-recon"],
            "biomevae-embed (base VAE)",
        )
        _check_embed_outputs(emb_dir)
        assert os.path.isfile(os.path.join(emb_dir, "recon.tsv"))

    def test_evaluate(self, synthetic_data, tmp_path):
        # First train
        model_dir = str(tmp_path / "base_vae_model2")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", model_dir] + FAST_TRAIN_ARGS,
            "train for test evaluation",
        )
        # Then test
        test_dir = str(tmp_path / "base_vae_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", test_dir,
             "--device", "cpu", "--export"],
            "biomevae-test (base VAE)",
        )
        _check_test_outputs(test_dir)


class TestVanillaVAE:
    """biomevae-train-vanilla"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "vanilla_vae")
        _run(
            ["biomevae-train-vanilla", "--input", synthetic_data["sgb"],
             "--outdir", outdir] + FAST_TRAIN_ARGS,
            "biomevae-train-vanilla",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("objective") == "vanilla"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "vanilla_model")
        _run(
            ["biomevae-train-vanilla", "--input", synthetic_data["sgb"],
             "--outdir", model_dir] + FAST_TRAIN_ARGS,
            "train vanilla for embed",
        )
        emb_dir = str(tmp_path / "vanilla_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", emb_dir,
             "--device", "cpu", "--export-recon"],
            "biomevae-embed (vanilla)",
        )
        _check_embed_outputs(emb_dir)
        assert os.path.isfile(os.path.join(emb_dir, "recon.tsv"))

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "vanilla_model2")
        _run(
            ["biomevae-train-vanilla", "--input", synthetic_data["sgb"],
             "--outdir", model_dir] + FAST_TRAIN_ARGS,
            "train vanilla for test",
        )
        test_dir = str(tmp_path / "vanilla_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", test_dir,
             "--device", "cpu", "--export"],
            "biomevae-test (vanilla)",
        )
        _check_test_outputs(test_dir)


@pytest.mark.skipif(not _has_geoopt, reason="geoopt not installed")
class TestHyperbolicVAE:
    """biomevae-train-hyp"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "hyp_vae")
        _run(
            ["biomevae-train-hyp", "--input", synthetic_data["sgb"],
             "--outdir", outdir,
             "--curvature", "1.0"] + FAST_TRAIN_ARGS,
            "biomevae-train-hyp",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "hyperbolic"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hyp_model")
        _run(
            ["biomevae-train-hyp", "--input", synthetic_data["sgb"],
             "--outdir", model_dir,
             "--curvature", "1.0"] + FAST_TRAIN_ARGS,
            "train hyp for embed",
        )
        emb_dir = str(tmp_path / "hyp_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", emb_dir,
             "--device", "cpu", "--emb-space", "ball"],
            "biomevae-embed (hyperbolic, ball)",
        )
        _check_embed_outputs(emb_dir)

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hyp_model2")
        _run(
            ["biomevae-train-hyp", "--input", synthetic_data["sgb"],
             "--outdir", model_dir,
             "--curvature", "1.0"] + FAST_TRAIN_ARGS,
            "train hyp for test",
        )
        test_dir = str(tmp_path / "hyp_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", test_dir,
             "--device", "cpu", "--export"],
            "biomevae-test (hyperbolic)",
        )
        _check_test_outputs(test_dir)


class TestTaxAwareVAE:
    """biomevae-train-tax"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "tax_vae")
        _run(
            ["biomevae-train-tax", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--tax-loss-levels", "g", "f",
             "--tax-loss-weight", "0.2"] + FAST_TRAIN_ARGS,
            "biomevae-train-tax",
        )
        _check_training_outputs(outdir)

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "tax_model")
        _run(
            ["biomevae-train-tax", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--tax-loss-levels", "g", "f",
             "--tax-loss-weight", "0.2"] + FAST_TRAIN_ARGS,
            "train tax for embed",
        )
        emb_dir = str(tmp_path / "tax_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", emb_dir,
             "--device", "cpu", "--export-recon"],
            "biomevae-embed (tax-aware)",
        )
        _check_embed_outputs(emb_dir)

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "tax_model2")
        _run(
            ["biomevae-train-tax", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--tax-loss-levels", "g", "f",
             "--tax-loss-weight", "0.2"] + FAST_TRAIN_ARGS,
            "train tax for test",
        )
        test_dir = str(tmp_path / "tax_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", test_dir,
             "--device", "cpu", "--export"],
            "biomevae-test (tax-aware)",
        )
        _check_test_outputs(test_dir)


@pytest.mark.skipif(not _has_geoopt, reason="geoopt not installed")
class TestHypTaxVAE:
    """biomevae-train-hyp-tax"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "hyptax_vae")
        _run(
            ["biomevae-train-hyp-tax", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--curvature", "1.0",
             "--tax-loss-levels", "g", "f",
             "--tax-loss-weight", "0.2"] + FAST_TRAIN_ARGS,
            "biomevae-train-hyp-tax",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "hyperbolic"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hyptax_model")
        _run(
            ["biomevae-train-hyp-tax", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--curvature", "1.0",
             "--tax-loss-levels", "g", "f",
             "--tax-loss-weight", "0.2"] + FAST_TRAIN_ARGS,
            "train hyp-tax for embed",
        )
        emb_dir = str(tmp_path / "hyptax_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", emb_dir,
             "--device", "cpu", "--emb-space", "ball", "--export-recon"],
            "biomevae-embed (hyp-tax, ball)",
        )
        _check_embed_outputs(emb_dir)

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hyptax_model2")
        _run(
            ["biomevae-train-hyp-tax", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--curvature", "1.0",
             "--tax-loss-levels", "g", "f",
             "--tax-loss-weight", "0.2"] + FAST_TRAIN_ARGS,
            "train hyp-tax for test",
        )
        test_dir = str(tmp_path / "hyptax_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir, "--outdir", test_dir,
             "--device", "cpu", "--export"],
            "biomevae-test (hyp-tax)",
        )
        _check_test_outputs(test_dir)


class TestGraphVAE:
    """biomevae-train-graph"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "graph_vae")
        _run(
            ["biomevae-train-graph", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--gnn", "gcn",
             "--gnn-hidden", "16",
             "--gnn-layers", "2"] + FAST_TRAIN_ARGS,
            "biomevae-train-graph",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "graph_tax"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "graph_model")
        _run(
            ["biomevae-train-graph", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--gnn", "gcn", "--gnn-hidden", "16", "--gnn-layers", "2"] + FAST_TRAIN_ARGS,
            "train graph for embed",
        )
        emb_dir = str(tmp_path / "graph_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu"],
            "biomevae-embed (graph)",
        )
        _check_embed_outputs(emb_dir)

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "graph_model2")
        _run(
            ["biomevae-train-graph", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--gnn", "gcn", "--gnn-hidden", "16", "--gnn-layers", "2"] + FAST_TRAIN_ARGS,
            "train graph for test",
        )
        test_dir = str(tmp_path / "graph_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", test_dir, "--device", "cpu", "--export"],
            "biomevae-test (graph)",
        )
        _check_test_outputs(test_dir)


class TestTreePriorVAE:
    """biomevae-train-treeprior"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "treeprior_vae")
        _run(
            ["biomevae-train-treeprior", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--prior", "brownian",
             "--prior-sigma", "1.0",
             "--gnn-hidden", "16",
             "--gnn-layers", "2"] + FAST_TRAIN_ARGS,
            "biomevae-train-treeprior",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "treeprior"

    def test_embed_and_test(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "tp_model")
        _run(
            ["biomevae-train-treeprior", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--prior", "brownian", "--prior-sigma", "1.0",
             "--gnn-hidden", "16", "--gnn-layers", "2"] + FAST_TRAIN_ARGS,
            "train treeprior for embed+test",
        )
        emb_dir = str(tmp_path / "tp_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu"],
            "biomevae-embed (treeprior)",
        )
        _check_embed_outputs(emb_dir)

        test_dir = str(tmp_path / "tp_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", test_dir, "--device", "cpu", "--export"],
            "biomevae-test (treeprior)",
        )
        _check_test_outputs(test_dir)


class TestPhyloFusionVAE:
    """biomevae-train-fuse"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "fuse_vae")
        _run(
            ["biomevae-train-fuse", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--phylo-embed", "pca",
             "--phylo-embed-dim", "8"] + FAST_TRAIN_ARGS,
            "biomevae-train-fuse",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "phylo_fusion"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "fuse_model")
        _run(
            ["biomevae-train-fuse", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--phylo-embed", "pca", "--phylo-embed-dim", "8"] + FAST_TRAIN_ARGS,
            "train fuse for embed",
        )
        emb_dir = str(tmp_path / "fuse_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu"],
            "biomevae-embed (phylo_fusion)",
        )
        _check_embed_outputs(emb_dir)

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "fuse_model2")
        _run(
            ["biomevae-train-fuse", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--phylo-embed", "pca", "--phylo-embed-dim", "8"] + FAST_TRAIN_ARGS,
            "train fuse for test",
        )
        test_dir = str(tmp_path / "fuse_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", test_dir, "--device", "cpu", "--export"],
            "biomevae-test (phylo_fusion)",
        )
        _check_test_outputs(test_dir)


class TestFlowXFormerVAE:
    """biomevae-train-flowxformer"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "flowx_vae")
        _run(
            ["biomevae-train-flowxformer",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--epochs", "3",
             "--batch-size", "8",
             "--hidden", "32", "16",
             "--latent-dim", "4",
             "--lr", "1e-3",
             "--early-stop", "0",
             "--device", "cpu",
             "--log1p",
             "--seed", "42",
             "--d-model", "32",
             "--n-layers", "1",
             "--n-heads", "4",
             "--uot", "root_l1",
             "--uot-lambda", "0.1",
             "--consistency-weight", "0.5",
             "--geom-weight", "0.0",
             "--no-amp"],
            "biomevae-train-flowxformer",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "flowxformer"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "flowx_model")
        _run(
            ["biomevae-train-flowxformer",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--epochs", "3",
             "--batch-size", "8",
             "--hidden", "32", "16",
             "--latent-dim", "4",
             "--lr", "1e-3",
             "--early-stop", "0",
             "--device", "cpu",
             "--log1p", "--seed", "42",
             "--d-model", "32", "--n-layers", "1", "--n-heads", "4",
             "--uot", "root_l1", "--no-amp"],
            "train flowx for embed",
        )
        emb_dir = str(tmp_path / "flowx_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu",
             "--export-recon"],
            "biomevae-embed (flowxformer)",
        )
        _check_embed_outputs(emb_dir)


@pytest.mark.skipif(not _has_torch_geometric, reason="torch_geometric not installed")
class TestHGVAEZI:
    """biomevae-train-hgvae-zi"""

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "hgvae_zi")
        _run(
            ["biomevae-train-hgvae-zi",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--epochs", "5",
             "--batch-size", "8",
             "--hidden", "32",
             "--latent-dim", "3",
             "--lr", "1e-3",
             "--beta-max", "1.0",
             "--seed", "42",
             "--device", "cpu"],
            "biomevae-train-hgvae-zi",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "hgvae_zi"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hgvae_model")
        _run(
            ["biomevae-train-hgvae-zi",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--epochs", "3", "--batch-size", "8",
             "--hidden", "32", "--latent-dim", "3",
             "--lr", "1e-3", "--seed", "42", "--device", "cpu"],
            "train hgvae for embed",
        )
        emb_dir = str(tmp_path / "hgvae_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu",
             "--export-recon"],
            "biomevae-embed (hgvae_zi)",
        )
        _check_embed_outputs(emb_dir)

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hgvae_model2")
        _run(
            ["biomevae-train-hgvae-zi",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir,
             "--epochs", "3", "--batch-size", "8",
             "--hidden", "32", "--latent-dim", "3",
             "--lr", "1e-3", "--seed", "42", "--device", "cpu"],
            "train hgvae for test",
        )
        test_dir = str(tmp_path / "hgvae_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", test_dir, "--device", "cpu", "--export"],
            "biomevae-test (hgvae_zi)",
        )
        _check_test_outputs(test_dir)


class TestPhILRVAE:
    """biomevae-train-philrvae"""

    PHILR_TRAIN_ARGS = [
        "--epochs", "5",
        "--batch-size", "8",
        "--hidden", "32", "16",
        "--latent-dim", "4",
        "--lr", "1e-3",
        "--early-stop", "0",
        "--device", "cpu",
        "--seed", "42",
    ]

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "philr_vae")
        _run(
            ["biomevae-train-philrvae",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir] + self.PHILR_TRAIN_ARGS,
            "biomevae-train-philrvae",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "philrvae"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "philr_model")
        _run(
            ["biomevae-train-philrvae",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir] + self.PHILR_TRAIN_ARGS,
            "train philr for embed",
        )
        emb_dir = str(tmp_path / "philr_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu",
             "--export-recon"],
            "biomevae-embed (philrvae)",
        )
        _check_embed_outputs(emb_dir)
        assert os.path.isfile(os.path.join(emb_dir, "recon.tsv"))

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "philr_model2")
        _run(
            ["biomevae-train-philrvae",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir] + self.PHILR_TRAIN_ARGS,
            "train philr for test",
        )
        test_dir = str(tmp_path / "philr_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", test_dir, "--device", "cpu", "--export"],
            "biomevae-test (philrvae)",
        )
        _check_test_outputs(test_dir)


@pytest.mark.skipif(not _has_geoopt, reason="geoopt not installed")
class TestHyperbolicPhILRVAE:
    """biomevae-train-hyp-philrvae — Poincaré-ball latent + PhILR + NB."""

    TRAIN_ARGS = [
        "--epochs", "5",
        "--batch-size", "8",
        "--hidden", "32", "16",
        "--latent-dim", "4",
        "--lr", "1e-3",
        "--curvature", "1.0",
        "--early-stop", "0",
        "--device", "cpu",
        "--seed", "42",
    ]

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "hyp_philrvae")
        _run(
            ["biomevae-train-hyp-philrvae",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir] + self.TRAIN_ARGS,
            "biomevae-train-hyp-philrvae",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "hyperbolic-philrvae"
        assert float(cfg.get("curvature", 0.0)) > 0.0

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hyp_philr_model")
        _run(
            ["biomevae-train-hyp-philrvae",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir] + self.TRAIN_ARGS,
            "train hyp-philrvae for embed",
        )
        emb_dir = str(tmp_path / "hyp_philr_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu",
             "--export-recon"],
            "biomevae-embed (hyp-philrvae)",
        )
        _check_embed_outputs(emb_dir)
        assert os.path.isfile(os.path.join(emb_dir, "recon.tsv"))

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "hyp_philr_model2")
        _run(
            ["biomevae-train-hyp-philrvae",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir] + self.TRAIN_ARGS,
            "train hyp-philrvae for test",
        )
        test_dir = str(tmp_path / "hyp_philr_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", test_dir, "--device", "cpu", "--export"],
            "biomevae-test (hyp-philrvae)",
        )
        _check_test_outputs(test_dir)



class TestTreeDTMVAE:
    """biomevae-train-tree-dtm"""

    TREE_DMT_TRAIN_ARGS = [
        "--epochs", "5",
        "--batch-size", "8",
        "--hidden", "32",
        "--latent-dim", "3",
        "--encoder-layers", "2",
        "--decoder-hidden", "32",
        "--decoder-layers", "2",
        "--lr", "1e-3",
        "--seed", "42",
        "--device", "cpu",
        "--data-kind", "counts",
        "--likelihood", "dirichlet_tree_multinomial",
    ]

    def test_train(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "tree-dtm-vae")
        _run(
            ["biomevae-train-tree-dtm",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir] + self.TREE_DMT_TRAIN_ARGS,
            "biomevae-train-tree-dtm",
        )
        _check_training_outputs(outdir)
        cfg = json.loads(Path(os.path.join(outdir, "config.json")).read_text())
        assert cfg.get("model_type") == "tree-dtm-vae"

    def test_embed(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "tree_dmt_model")
        _run(
            ["biomevae-train-tree-dtm",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir] + self.TREE_DMT_TRAIN_ARGS,
            "train tree-dmt for embed",
        )
        emb_dir = str(tmp_path / "tree_dmt_embed")
        _run(
            ["biomevae-embed", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", emb_dir, "--device", "cpu",
             "--export-recon"],
            "biomevae-embed (tree-dtm-vae)",
        )
        _check_embed_outputs(emb_dir)
        assert os.path.isfile(os.path.join(emb_dir, "recon.tsv"))

    def test_evaluate(self, synthetic_data, tmp_path):
        model_dir = str(tmp_path / "tree_dmt_model2")
        _run(
            ["biomevae-train-tree-dtm",
             "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", model_dir] + self.TREE_DMT_TRAIN_ARGS,
            "train tree-dmt for test",
        )
        test_dir = str(tmp_path / "tree_dmt_test")
        _run(
            ["biomevae-test", "--input", synthetic_data["sgb"],
             "--model-dir", model_dir,
             "--taxonomy", synthetic_data["tax"],
             "--outdir", test_dir, "--device", "cpu", "--export"],
            "biomevae-test (tree-dtm-vae)",
        )
        _check_test_outputs(test_dir)

class TestNMFBaseline:
    """biomevae-nmf"""

    def test_nmf_fixed_rank(self, synthetic_data, tmp_path):
        out_json = str(tmp_path / "nmf_result.json")
        _run(
            ["biomevae-nmf", "--input", synthetic_data["sgb"],
             "--components", "4", "--splits", "2",
             "--seed", "42", "--output", out_json],
            "biomevae-nmf (fixed rank=4)",
        )
        assert os.path.isfile(out_json)
        result = json.loads(Path(out_json).read_text())
        assert "mae_mean" in result or "mean_mae" in result or isinstance(result, dict)

    def test_nmf_rank_selection(self, synthetic_data, tmp_path):
        out_json = str(tmp_path / "nmf_rank_sel.json")
        _run(
            ["biomevae-nmf", "--input", synthetic_data["sgb"],
             "--rank-candidates", "2,4,6", "--splits", "2",
             "--seed", "42", "--output", out_json],
            "biomevae-nmf (rank selection)",
        )
        assert os.path.isfile(out_json)


class TestCompareToNMF:
    """biomevae-comparetonmf"""

    def test_compare(self, synthetic_data, tmp_path):
        # First train a base model to get its config
        model_dir = str(tmp_path / "compare_model")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", model_dir] + FAST_TRAIN_ARGS,
            "train model for comparison",
        )
        out_json = str(tmp_path / "compare_result.json")
        _run(
            ["biomevae-comparetonmf", "--input", synthetic_data["sgb"],
             "--method-name", "beta-vae",
             "--method-config", os.path.join(model_dir, "config.json"),
             "--components", "4", "--splits", "2",
             "--seed", "42", "--device", "cpu",
             "--output", out_json],
            "biomevae-comparetonmf",
        )
        assert os.path.isfile(out_json)


class TestObjectiveVariants:
    """Test different objective functions on base VAE."""

    def test_capacity_objective(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "cap_vae")
        # biomevae-train locks objective to beta; use biomevae-train-tax which
        # accepts --objective and doesn't require optional dependencies
        _run(
            ["biomevae-train-tax", "--input", synthetic_data["sgb"],
             "--taxonomy", synthetic_data["tax"],
             "--outdir", outdir,
             "--objective", "capacity",
             "--capacity-gamma", "1.0",
             "--capacity-epochs", "3",
             "--tax-loss-levels", "g",
             "--tax-loss-weight", "0.0"] + FAST_TRAIN_ARGS,
            "biomevae-train-tax (capacity objective)",
        )
        _check_training_outputs(outdir)

    def test_huber_loss(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "huber_vae")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", outdir,
             "--recon", "huber", "--huber-delta", "0.5"] + FAST_TRAIN_ARGS,
            "biomevae-train (huber recon loss)",
        )
        _check_training_outputs(outdir)

    def test_mae_loss(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "mae_vae")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", outdir,
             "--recon", "mae"] + FAST_TRAIN_ARGS,
            "biomevae-train (mae recon loss)",
        )
        _check_training_outputs(outdir)

    def test_standardize(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "std_vae")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", outdir,
             "--standardize"] + FAST_TRAIN_ARGS,
            "biomevae-train (with standardization)",
        )
        _check_training_outputs(outdir)
        assert os.path.isfile(os.path.join(outdir, "feature_scaler.npz"))

    def test_layer_norm_gelu(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "ln_vae")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", outdir,
             "--layer-norm", "--activation", "gelu"] + FAST_TRAIN_ARGS,
            "biomevae-train (layer-norm + gelu)",
        )
        _check_training_outputs(outdir)

    def test_adamw_with_weight_decay(self, synthetic_data, tmp_path):
        outdir = str(tmp_path / "adamw_vae")
        _run(
            ["biomevae-train", "--input", synthetic_data["sgb"],
             "--outdir", outdir,
             "--optimizer", "adamw",
             "--weight-decay", "1e-4"] + FAST_TRAIN_ARGS,
            "biomevae-train (adamw + weight decay)",
        )
        _check_training_outputs(outdir)
