"""Unit tests for DS-VAE (Disease-Supervised Phylogenetic VAE).

Covers:

* Shape contracts on ``forward`` for both variants.
* Class-conditional prior behaviour (KL → 0 when the encoder matches).
* Orthogonality of the class-mean initialisation.
* Loss helpers (focal CE, SupCon, cyclical β, gaussian_kl).
* MixUp soft-label + coords arithmetic.
* End-to-end tiny training smoke test through ``train_once_dsvae``.
* Snakemake dry-run covering both ``dsvae-unsup`` and ``dsvae-sup`` rules.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biomevae.losses import (
    cyclical_beta_schedule,
    effective_number_class_weights,
    focal_ce_balanced,
    gaussian_kl,
    supcon_loss,
)
from biomevae.models.dsvae import (
    ClassConditionalPrior,
    DSVAE,
    _orthogonal_frame,
    philr_mixup,
)
from biomevae.models.tree_spec import TreeSpec
from biomevae.trainers.train_loop import train_once_dsvae


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _tiny_tree_spec(n_leaves: int = 12) -> TreeSpec:
    """Build a tiny balanced binary tree spec via ``build_tree_spec``.

    The test keeps everything self-contained: we synthesise a phyla.tsv
    with n_leaves clades distributed across a small taxonomy so that
    build_tree_spec returns a non-trivial (n_leaves, n_leaves-1)
    contrast.
    """
    from biomevae.models.tree_spec import build_tree_spec

    tmp = tempfile.TemporaryDirectory()
    tax_path = Path(tmp.name) / "phyla.tsv"
    rows = []
    for i in range(n_leaves):
        clade = f"t__SGB{i:03d}"
        rank_ids = [
            f"k__King{i % 2}",
            f"p__Phylum{i % 3}",
            f"c__Class{i % 4}",
            f"o__Order{i % 2}",
            f"f__Family{i % 3}",
            f"g__Genus{i % 4}",
            f"s__Species{i}",
        ]
        rows.append("\t".join([clade] + rank_ids))
    tax_path.write_text("\n".join(rows) + "\n")
    clades = [f"t__SGB{i:03d}" for i in range(n_leaves)]
    spec = build_tree_spec(clades, str(tax_path), branchlen_mode="unit")
    # Keep the tmp dir alive on the returned spec for the duration of
    # the test session.
    spec._tmpdir = tmp  # type: ignore[attr-defined]
    return spec


@pytest.fixture(scope="module")
def tree_spec() -> TreeSpec:
    return _tiny_tree_spec(n_leaves=12)


@pytest.fixture(scope="module")
def synthetic_counts() -> np.ndarray:
    rng = np.random.RandomState(0)
    # (n_samples, n_features) = (16, 12)
    return rng.poisson(lam=3.0, size=(16, 12)).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_labels() -> np.ndarray:
    return np.array([i % 3 for i in range(16)], dtype=np.int64)


# ---------------------------------------------------------------------------
# 1. Shape contracts
# ---------------------------------------------------------------------------


def test_dsvae_forward_shapes(tree_spec):
    model = DSVAE(n_features=12, latent_dim=8, tree_spec=tree_spec)
    x = torch.randn(4, 12).abs()
    mu_x, mu_z, logvar_z = model(x)
    assert mu_x.shape == (4, 12)
    assert mu_z.shape == (4, 8)
    assert logvar_z.shape == (4, 8)
    # Decoded output must be strictly positive (count-space mean).
    assert torch.all(mu_x > 0)


def test_dsvae_supervised_requires_n_classes(tree_spec):
    with pytest.raises(ValueError):
        DSVAE(
            n_features=12, latent_dim=8, tree_spec=tree_spec,
            supervised=True,
        )


def test_dsvae_classify_forward(tree_spec):
    model = DSVAE(
        n_features=12, latent_dim=8, tree_spec=tree_spec,
        supervised=True, n_classes=3,
    )
    x = torch.randn(4, 12).abs()
    _, mu_z, _ = model(x)
    logits = model.classify(mu_z)
    assert logits.shape == (4, 3)


# ---------------------------------------------------------------------------
# 2. Class-conditional prior
# ---------------------------------------------------------------------------


def test_orthogonal_frame_is_unit_norm():
    frame = _orthogonal_frame(n_classes=5, dim=8, scale=1.0)
    norms = frame.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_class_conditional_prior_kl_matches_posterior():
    """KL(q‖p) = 0 when q == p (same μ, logvar)."""
    prior = ClassConditionalPrior(n_classes=4, latent_dim=6)
    y = torch.tensor([0, 1, 2, 3])
    mu_p, logvar_p = prior(y)
    kl = gaussian_kl(mu_p, logvar_p, mu_p, logvar_p).mean()
    assert kl.abs() < 1e-5


def test_class_prior_deterministic_across_instances():
    p1 = ClassConditionalPrior(n_classes=4, latent_dim=6)
    p2 = ClassConditionalPrior(n_classes=4, latent_dim=6)
    # Same seed → same init.
    assert torch.allclose(p1.mu, p2.mu)


# ---------------------------------------------------------------------------
# 3. Loss helpers
# ---------------------------------------------------------------------------


def test_cyclical_beta_schedule_shape():
    # 4 cycles of length 50 with 50% linear ramp → at cycle start β = 0,
    # at half-cycle β = beta_max, and it stays saturated thereafter.
    assert cyclical_beta_schedule(1, n_cycles=4, cycle_len=50) == 0.0
    mid = cyclical_beta_schedule(25, n_cycles=4, cycle_len=50, beta_max=1.0)
    assert 0.0 < mid <= 1.0
    assert cyclical_beta_schedule(26, n_cycles=4, cycle_len=50) == 1.0
    # Past total length → pinned.
    assert cyclical_beta_schedule(1000, n_cycles=4, cycle_len=50) == 1.0


def test_focal_ce_balanced_hard_and_soft():
    logits = torch.randn(4, 3)
    hard = torch.tensor([0, 1, 2, 1])
    soft = torch.nn.functional.one_hot(hard, num_classes=3).float()
    loss_hard = focal_ce_balanced(logits, hard)
    loss_soft = focal_ce_balanced(logits, soft)
    # Hard and soft should agree on a pure one-hot.
    assert torch.allclose(loss_hard, loss_soft, atol=1e-5)


def test_focal_ce_class_weight_scales():
    logits = torch.randn(8, 3)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    loss_a = focal_ce_balanced(logits, y)
    loss_b = focal_ce_balanced(
        logits, y, class_weight=torch.tensor([1.0, 2.0, 2.0]),
    )
    assert not torch.allclose(loss_a, loss_b)


def test_supcon_loss_positive_and_no_positive():
    # No positive pairs → zero.
    feats = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    labels = torch.tensor([0, 1, 2])
    assert supcon_loss(feats, labels).abs() < 1e-7
    # With positive pairs → positive scalar.
    feats = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    assert supcon_loss(feats, labels).item() > 0.0


def test_effective_number_weights_sum():
    counts = torch.tensor([10.0, 100.0, 1000.0])
    w = effective_number_class_weights(counts)
    assert torch.isfinite(w).all()
    assert abs(float(w.sum()) - 3.0) < 1e-4


# ---------------------------------------------------------------------------
# 4. PhILR-space MixUp
# ---------------------------------------------------------------------------


def test_philr_mixup_bounds():
    coords = torch.randn(4, 8)
    y = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2, 0]), num_classes=3,
    ).float()
    mixed, mixed_y, lam = philr_mixup(coords, y, alpha=0.5)
    # Mixed coords inside the convex hull of (coords, coords[perm]).
    assert mixed.shape == coords.shape
    assert mixed_y.shape == y.shape
    assert 0.0 <= float(lam) <= 1.0
    # Soft labels should still sum to 1 per sample.
    assert torch.allclose(mixed_y.sum(dim=-1), torch.ones(4))


def test_philr_mixup_alpha_zero_is_noop():
    coords = torch.randn(3, 4)
    y = torch.eye(3)
    mixed, mixed_y, lam = philr_mixup(coords, y, alpha=0.0)
    assert torch.allclose(mixed, coords)
    assert torch.allclose(mixed_y, y)
    assert float(lam) == 1.0


# ---------------------------------------------------------------------------
# 5. Smoke tests through train_once_dsvae
# ---------------------------------------------------------------------------


def _base_train_params(supervised: bool) -> dict:
    return {
        "device": "cpu",
        "model_type": "dsvae",
        "supervised": supervised,
        "latent_dim": 4,
        "hidden": [32, 16],
        "dropout": 0.0,
        "pseudocount": 0.5,
        "classifier_hidden": 16,
        "branchlen_mode": "unit",
        "epochs": 3,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "beta_max": 1.0,
        "beta_n_cycles": 1,
        "beta_cycle_len": 2,
        "beta_ramp_frac": 0.5,
        "free_bits": 0.0,
        "gamma_cls": 1.0,
        "gamma_con": 0.3,
        "focal_gamma": 2.0,
        "supcon_tau": 0.1,
        "mixup_alpha": 0.0,
        "effnum_beta": 0.9999,
        "val_split": 0.25,
        "early_stop": 0,
    }


def test_train_unsupervised_runs(tmp_path, tree_spec, synthetic_counts):
    params = _base_train_params(supervised=False)
    # Pass the already-built tree_spec through the serialised form to
    # skip re-parsing the taxonomy.
    params["tree_spec"] = tree_spec.to_json()

    res = train_once_dsvae(
        synthetic_counts,
        sample_names=[f"s{i}" for i in range(synthetic_counts.shape[0])],
        outdir=str(tmp_path / "unsup"),
        params=params,
        seed=0,
        verbose=False,
        return_model=True,
    )
    assert np.isfinite(res["val_recon"])
    assert res["model"] is not None
    assert os.path.isfile(str(tmp_path / "unsup" / "model.pt"))
    assert os.path.isfile(str(tmp_path / "unsup" / "training_log.tsv"))


def test_train_supervised_runs(tmp_path, tree_spec, synthetic_counts, synthetic_labels):
    params = _base_train_params(supervised=True)
    params["tree_spec"] = tree_spec.to_json()
    params["n_classes"] = int(synthetic_labels.max()) + 1

    res = train_once_dsvae(
        synthetic_counts,
        sample_names=[f"s{i}" for i in range(synthetic_counts.shape[0])],
        outdir=str(tmp_path / "sup"),
        params=params,
        seed=0,
        verbose=False,
        return_model=True,
        labels=synthetic_labels,
    )
    # Training completed, val metrics reported.
    assert "val_macro_f1" in res
    assert "val_balanced_accuracy" in res
    assert os.path.isfile(str(tmp_path / "sup" / "model.pt"))


# ---------------------------------------------------------------------------
# 6. CLI smoke test (unsupervised variant, end to end through the
#    registered console-script entry point).
# ---------------------------------------------------------------------------


def _write_tiny_sgb(tmp_path: Path, n_taxa: int = 12, n_samples: int = 16) -> Path:
    rng = np.random.RandomState(1)
    clades = [f"t__SGB{i:03d}" for i in range(n_taxa)]
    tax_ids = [f"{1000 + i}" for i in range(n_taxa)]
    samples = [f"sample_{i:03d}" for i in range(n_samples)]
    counts = rng.poisson(lam=3.0, size=(n_taxa, n_samples)).astype(int)
    rows = []
    header = "\t".join(["clade_name", "NCBI_tax_id"] + samples)
    rows.append(header)
    for i in range(n_taxa):
        rows.append("\t".join(
            [clades[i], tax_ids[i]] + [str(int(v)) for v in counts[i]]
        ))
    path = tmp_path / "sgb_table.tsv"
    path.write_text("\n".join(rows) + "\n")
    return path


def _write_tiny_phyla(tmp_path: Path, n_taxa: int = 12) -> Path:
    rows = []
    for i in range(n_taxa):
        rows.append("\t".join([
            f"t__SGB{i:03d}",
            f"k__King{i % 2}",
            f"p__Phylum{i % 3}",
            f"c__Class{i % 4}",
            f"o__Order{i % 2}",
            f"f__Family{i % 3}",
            f"g__Genus{i % 4}",
            f"s__Species{i}",
        ]))
    path = tmp_path / "phyla.tsv"
    path.write_text("\n".join(rows) + "\n")
    return path


def _write_tiny_metadata(tmp_path: Path, n_samples: int = 16) -> Path:
    path = tmp_path / "sample_metadata.tsv"
    with path.open("w") as fh:
        fh.write("sample_id\tdisease\n")
        for i in range(n_samples):
            fh.write(f"sample_{i:03d}\t{'case' if i % 2 == 0 else 'ctrl'}\n")
    return path


@pytest.mark.skipif(
    shutil.which("biomevae-train-dsvae") is None,
    reason="biomevae-train-dsvae CLI not installed (pip install -e .)",
)
def test_cli_unsupervised(tmp_path):
    sgb = _write_tiny_sgb(tmp_path)
    phyla = _write_tiny_phyla(tmp_path)
    outdir = tmp_path / "unsup_cli"
    proc = subprocess.run(
        [
            "biomevae-train-dsvae",
            "--input", str(sgb),
            "--taxonomy", str(phyla),
            "--outdir", str(outdir),
            "--no-supervised",
            "--epochs", "2",
            "--batch-size", "4",
            "--latent-dim", "4",
            "--hidden", "16",
            "--beta-cycle-len", "2",
            "--beta-n-cycles", "1",
            "--free-bits", "0.0",
            "--early-stop", "0",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert (outdir / "config.json").is_file()
    assert (outdir / "embeddings.tsv").is_file()
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == "dsvae"
    assert cfg["supervised"] is False


@pytest.mark.skipif(
    shutil.which("biomevae-train-dsvae") is None,
    reason="biomevae-train-dsvae CLI not installed (pip install -e .)",
)
def test_cli_supervised(tmp_path):
    sgb = _write_tiny_sgb(tmp_path)
    phyla = _write_tiny_phyla(tmp_path)
    meta = _write_tiny_metadata(tmp_path)
    outdir = tmp_path / "sup_cli"
    proc = subprocess.run(
        [
            "biomevae-train-dsvae",
            "--input", str(sgb),
            "--taxonomy", str(phyla),
            "--outdir", str(outdir),
            "--supervised",
            "--metadata", str(meta),
            "--label-col", "disease",
            "--epochs", "2",
            "--batch-size", "4",
            "--latent-dim", "4",
            "--hidden", "16",
            "--beta-cycle-len", "2",
            "--beta-n-cycles", "1",
            "--free-bits", "0.0",
            "--mixup-alpha", "0.0",
            "--early-stop", "0",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["model_type"] == "dsvae"
    assert cfg["supervised"] is True
    assert cfg.get("class_names") == ["case", "ctrl"]
    assert cfg.get("n_classes") == 2


# ---------------------------------------------------------------------------
# 7. Snakemake dry-run (both DS-VAE rules)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("snakemake") is None,
    reason="snakemake not installed",
)
def test_snakemake_dry_run_dsvae(tmp_path):
    # Build a tiny data_root with the three required files so
    # resolve_studies() accepts the study.
    data_root = tmp_path / "data"
    study_dir = data_root / "tiny_study"
    study_dir.mkdir(parents=True)
    _write_tiny_sgb(study_dir)
    _write_tiny_phyla(study_dir)
    _write_tiny_metadata(study_dir)

    output_root = tmp_path / "out"
    output_root.mkdir()

    repo_root = Path(__file__).resolve().parents[1]
    snakefile = repo_root / "workflow" / "Snakefile"
    if not snakefile.is_file():
        pytest.skip(f"Snakefile not found at {snakefile}")

    proc = subprocess.run(
        [
            "snakemake",
            "--snakefile", str(snakefile),
            "--directory", str(tmp_path),
            "--config",
            f"data_root={data_root}",
            f"output_root={output_root}",
            "study=tiny_study",
            "--forceall",
            "--dry-run",
            "--quiet",
            # Limit to just the train_model rule so we don't trigger
            # heavy downstream targets that aren't part of this test.
            f"{output_root}/tiny_study/models/dsvae-unsup/model.pt",
            f"{output_root}/tiny_study/models/dsvae-sup/model.pt",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # dry-run either exits 0 (all rules satisfiable) or prints a useful
    # error. A non-zero exit here is a test failure.
    if proc.returncode != 0:
        pytest.fail(
            "Snakemake dry-run failed for DS-VAE rules.\n"
            f"STDOUT:\n{proc.stdout[-1500:]}\n"
            f"STDERR:\n{proc.stderr[-1500:]}"
        )
