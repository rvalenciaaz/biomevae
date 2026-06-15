"""End-to-end smoke test for the single-study CAPDA-VAE.

Exercises the exact contract the standard / meta Snakemake pipeline relies on:

    train  -> model.pt + config.json + oof_embeddings.tsv
    test   -> test/test_report.json + embeddings/recon
    embed  -> embed/embeddings.tsv (leak-free OOF passthrough) + recon.tsv
    classify on embed/embeddings.tsv

so that ``capda-vae`` can be added to the single-study model catalogue and flow
through train/postprocess/classify/figures/aggregate like every other model.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomevae.models.capda_vae import capda_fit_single_study, load_lineage_table


def _write_fixture(tmp_path: Path, n_taxa: int = 14, n_samples: int = 40):
    rng = np.random.RandomState(0)
    clades = [f"t__SGB{i:03d}" for i in range(n_taxa)]
    samples = [f"sample_{i:03d}" for i in range(n_samples)]
    y = np.array([i % 2 for i in range(n_samples)])
    counts = rng.poisson(lam=3.0, size=(n_taxa, n_samples)).astype(int)
    counts[0] += y * 8         # case-enriched clade
    counts[1] += (1 - y) * 6   # control-enriched clade

    rows = ["\t".join(["clade_name", "NCBI_tax_id"] + samples)]
    for i in range(n_taxa):
        rows.append("\t".join([clades[i], str(1000 + i)]
                              + [str(int(v)) for v in counts[i]]))
    sgb = tmp_path / "sgb_table.tsv"
    sgb.write_text("\n".join(rows) + "\n")

    prows = []
    for i in range(n_taxa):
        prows.append("\t".join([
            clades[i], f"k__K{i % 2}", f"p__P{i % 3}", f"c__C{i % 4}",
            f"o__O{i % 2}", f"f__F{i % 3}", f"g__G{i % 4}", f"s__S{i}",
        ]))
    phyla = tmp_path / "phyla.tsv"
    phyla.write_text("\n".join(prows) + "\n")

    meta = tmp_path / "sample_metadata.tsv"
    with meta.open("w") as fh:
        fh.write("sample_id\tdisease\n")
        for i in range(n_samples):
            fh.write(f"{samples[i]}\t{'case' if y[i] == 1 else 'ctrl'}\n")
    return sgb, phyla, meta


def test_capda_single_study_fit_is_leak_free():
    """The fit returns leak-free OOF columns and a re-loadable config."""
    rng = np.random.RandomState(0)
    n_taxa, n_samples = 14, 40
    feat_clades = [f"t__SGB{i:03d}" for i in range(n_taxa)]
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    y = np.array([i % 2 for i in range(n_samples)])
    X = rng.poisson(3.0, size=(n_samples, n_taxa)).astype(np.float32)
    X[:, 0] += y * 8
    tax = pd.DataFrame(
        {lvl: [f"{lvl}__{i % 3}" for i in range(n_taxa)]
         for lvl in ["k", "p", "c", "o", "f", "g", "s"]},
        index=feat_clades,
    )
    y_raw = np.array(["case" if v == 1 else "ctrl" for v in y], dtype=object)

    df, state_dict, config = capda_fit_single_study(
        X, sample_ids, feat_clades, y_raw, tax,
        n_splits=3, seed=42, device="cpu",
        hp=dict(epochs=3, latent=4, hidden=16),
    )
    assert config["model_type"] == "capda-vae"
    assert config["single_study"] is True
    assert config["n_classes"] == 2
    assert df.shape == (n_samples, n_taxa + 2)
    # every sample got a proper probability row (OOF + final fallback)
    prob_cols = [c for c in df.columns if c.startswith("capda_prob_")]
    assert len(prob_cols) == 2
    probs = df[prob_cols].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.skipif(
    shutil.which("biomevae-train-capda-vae-ss") is None,
    reason="biomevae CLIs not installed (pip install -e .)",
)
def test_capda_single_study_cli_roundtrip(tmp_path):
    sgb, phyla, meta = _write_fixture(tmp_path)
    model_dir = tmp_path / "models" / "capda-vae"
    common = ["--device", "cpu"]

    # 1. train (matches the single-study train_model rule, incl. ignored args)
    proc = subprocess.run(
        ["biomevae-train-capda-vae-ss",
         "--input", str(sgb), "--taxonomy", str(phyla),
         "--metadata", str(meta), "--label-col", "disease",
         "--outdir", str(model_dir),
         "--vae-epochs", "3", "--latent", "4", "--hidden", "16",
         "--n-splits", "3", "--epochs", "100", "--optuna",
         "--optuna-trials", "100", *common],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (model_dir / "model.pt").is_file()
    assert (model_dir / "oof_embeddings.tsv").is_file()
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["model_type"] == "capda-vae"

    # 2. test --export (postprocess_test rule)
    test_dir = model_dir / "test"
    proc = subprocess.run(
        ["biomevae-test", "--input", str(sgb), "--model-dir", str(model_dir),
         "--outdir", str(test_dir), "--taxonomy", str(phyla), "--export",
         *common],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads((test_dir / "test_report.json").read_text())
    assert {"reconstruction", "kl_mean", "beta_loss_at_beta_max"} <= set(report)
    assert (test_dir / "embeddings.tsv").is_file()

    # 3. embed --export-recon (postprocess_embed rule) -> leak-free passthrough
    embed_dir = model_dir / "embed"
    proc = subprocess.run(
        ["biomevae-embed", "--input", str(sgb), "--model-dir", str(model_dir),
         "--outdir", str(embed_dir), "--taxonomy", str(phyla),
         "--export-recon", *common],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    emb = pd.read_csv(embed_dir / "embeddings.tsv", sep="\t", index_col=0)
    oof = pd.read_csv(model_dir / "oof_embeddings.tsv", sep="\t", index_col=0)
    assert list(emb.columns) == list(oof.columns)
    # embed must reproduce the stored leak-free OOF rows exactly
    assert np.allclose(emb.reindex(oof.index).to_numpy(),
                       oof.to_numpy(), atol=1e-5)
    assert (embed_dir / "recon.tsv").is_file()

    # 4. classify on the embed output (classify rule)
    classify_dir = model_dir / "classify"
    proc = subprocess.run(
        ["biomevae-classify", "--embeddings", str(embed_dir / "embeddings.tsv"),
         "--metadata", str(meta), "--label", "disease",
         "--outdir", str(classify_dir), "--n-splits", "3", "--n-repeats", "2",
         "--seeds", "42"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (classify_dir / "classification_results.json").is_file()
