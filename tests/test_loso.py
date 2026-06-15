"""Unit tests for ``biomevae.loso``: study merging, LOSO splitter, anchor diagnostics.

All tests use synthetic ``sgb_table.tsv`` / ``phyla.tsv`` /
``sample_metadata.tsv`` written into a tmp directory so they exercise
the on-disk I/O path used by the Snakemake rules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomevae.loso import (
    control_anchor,
    covariance_frobenius,
    leave_one_study_out_splits,
    load_merged,
    merge_studies,
    mmd_rbf_unbiased,
)


def _make_study(
    root: Path, name: str, sample_ids, clades, abund, disease,
):
    sdir = root / name
    sdir.mkdir(parents=True, exist_ok=True)

    # sgb_table.tsv
    cols = ["clade_name", "NCBI_tax_id"] + list(sample_ids)
    rows = []
    for ci, c in enumerate(clades):
        rows.append([c, str(1000 + ci)] + list(abund[ci]))
    pd.DataFrame(rows, columns=cols).to_csv(
        sdir / "sgb_table.tsv", sep="\t", index=False,
    )

    # phyla.tsv (no header by convention)
    phyla = pd.DataFrame([[c, "k__Bacteria", "p__Firmicutes"] for c in clades])
    phyla.to_csv(sdir / "phyla.tsv", sep="\t", index=False, header=False)

    # sample_metadata.tsv
    meta = pd.DataFrame({
        "sample_id": sample_ids,
        "disease": disease,
        "country": [f"C{i % 2}" for i in range(len(sample_ids))],
    })
    meta.to_csv(sdir / "sample_metadata.tsv", sep="\t", index=False)


def test_merge_studies_unifies_features_and_zero_fills(tmp_path):
    _make_study(
        tmp_path, "A",
        sample_ids=[f"A{i}" for i in range(3)],
        clades=["sgb1", "sgb2"],
        abund=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        disease=["healthy", "CRC", "healthy"],
    )
    _make_study(
        tmp_path, "B",
        sample_ids=[f"B{i}" for i in range(2)],
        clades=["sgb2", "sgb3"],
        abund=[[7.0, 8.0], [9.0, 10.0]],
        disease=["CRC", "healthy"],
    )
    merged = merge_studies(tmp_path, ["A", "B"])
    assert merged.feature_clades == ["sgb1", "sgb2", "sgb3"]
    assert sorted(merged.sample_to_study.keys()) == ["A0", "A1", "A2", "B0", "B1"]
    # B's zero-filled sgb1 row.
    sgb1_row = merged.sgb_table[merged.sgb_table["clade_name"] == "sgb1"]
    assert sgb1_row[["B0", "B1"]].astype(float).to_numpy().tolist() == [[0.0, 0.0]]
    # study_name column was injected into metadata.
    assert merged.metadata.loc["A0", "study_name"] == "A"
    assert merged.metadata.loc["B0", "study_name"] == "B"


def test_merge_studies_round_trips_via_disk(tmp_path):
    _make_study(
        tmp_path, "A",
        sample_ids=["A0", "A1"],
        clades=["sgb1", "sgb2"],
        abund=[[1.0, 2.0], [3.0, 4.0]],
        disease=["healthy", "CRC"],
    )
    _make_study(
        tmp_path, "B",
        sample_ids=["B0", "B1"],
        clades=["sgb2", "sgb3"],
        abund=[[5.0, 6.0], [7.0, 8.0]],
        disease=["CRC", "healthy"],
    )
    out = tmp_path / "_merged"
    merged = merge_studies(tmp_path, ["A", "B"])
    paths = merged.write(out)
    assert (out / "sgb_table.tsv").exists()
    assert (out / "phyla.tsv").exists()
    assert (out / "sample_metadata.tsv").exists()

    X, sample_ids, feature_clades, metadata = load_merged(
        paths["sgb_table"], paths["sample_metadata"],
    )
    assert sample_ids == ["A0", "A1", "B0", "B1"]
    assert feature_clades == ["sgb1", "sgb2", "sgb3"]
    assert X.shape == (4, 3)
    # A0/A1 should have zero in sgb3 (column index 2).
    assert X[0, 2] == 0.0 and X[1, 2] == 0.0
    # B0/B1 should have zero in sgb1 (column index 0).
    assert X[2, 0] == 0.0 and X[3, 0] == 0.0


def test_merge_rejects_duplicate_sample_ids(tmp_path):
    _make_study(
        tmp_path, "A",
        sample_ids=["dup", "A1"],
        clades=["sgb1"], abund=[[1.0, 2.0]], disease=["CRC", "healthy"],
    )
    _make_study(
        tmp_path, "B",
        sample_ids=["dup", "B1"],
        clades=["sgb1"], abund=[[3.0, 4.0]], disease=["CRC", "healthy"],
    )
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        merge_studies(tmp_path, ["A", "B"])


def test_loso_splitter_yields_one_fold_per_study():
    sample_ids = [f"s{i}" for i in range(10)]
    sample_to_study = {
        **{f"s{i}": "A" for i in range(0, 4)},
        **{f"s{i}": "B" for i in range(4, 7)},
        **{f"s{i}": "C" for i in range(7, 10)},
    }
    folds = list(leave_one_study_out_splits(sample_ids, sample_to_study))
    assert [held for _, _, held in folds] == ["A", "B", "C"]
    for tr, ev, held in folds:
        assert tr.sum() + ev.sum() == 10
        # No overlap.
        assert not np.any(tr & ev)
        # The eval set contains exactly the held-out study's samples.
        held_samples = {sid for sid, st in sample_to_study.items() if st == held}
        eval_samples = {sample_ids[i] for i, m in enumerate(ev) if m}
        assert held_samples == eval_samples


def test_loso_splitter_only_studies_filter():
    sample_ids = ["s0", "s1", "s2"]
    sample_to_study = {"s0": "A", "s1": "B", "s2": "C"}
    folds = list(leave_one_study_out_splits(
        sample_ids, sample_to_study, only_studies=["B"],
    ))
    assert len(folds) == 1
    assert folds[0][2] == "B"


def test_covariance_frobenius_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    assert covariance_frobenius(X, X) == pytest.approx(0.0, abs=1e-9)


def test_mmd_close_to_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    Y = rng.normal(size=(80, 4))
    val = mmd_rbf_unbiased(X, Y)
    # Same distribution → unbiased MMD² fluctuates around 0.
    assert abs(val) < 0.05


def test_mmd_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    Y = rng.normal(loc=3.0, size=(80, 4))
    val = mmd_rbf_unbiased(X, Y)
    assert val > 0.05


def test_control_anchor_selects_only_control_samples():
    rng = np.random.default_rng(0)
    n_per_study = 10
    Z = rng.normal(size=(2 * n_per_study, 3)).astype(np.float32)
    sample_ids = [f"s{i}" for i in range(2 * n_per_study)]
    metadata = pd.DataFrame({
        "study_name": ["A"] * n_per_study + ["B"] * n_per_study,
        # Half of each study is healthy, half is CRC.
        "disease": (["healthy"] * (n_per_study // 2)
                    + ["CRC"] * (n_per_study // 2)) * 2,
    }, index=sample_ids)

    anchor = control_anchor(
        Z, sample_ids, metadata,
        label_col="disease", control_value="healthy", min_samples=3,
    )
    assert anchor.studies == ["A", "B"]
    assert len(anchor.coral_pairs) == 1
    assert len(anchor.mmd_pairs) == 1
    assert anchor.coral_pairs.iloc[0]["n_a"] == n_per_study // 2
    assert anchor.coral_pairs.iloc[0]["n_b"] == n_per_study // 2


def test_control_anchor_skips_studies_below_min_samples():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(8, 2)).astype(np.float32)
    sample_ids = [f"s{i}" for i in range(8)]
    metadata = pd.DataFrame({
        "study_name": ["A", "A", "A", "A", "A", "B", "B", "C"],
        "disease": ["healthy"] * 5 + ["healthy", "healthy", "healthy"],
    }, index=sample_ids)

    anchor = control_anchor(
        Z, sample_ids, metadata,
        label_col="disease", control_value="healthy", min_samples=3,
    )
    # B has 2 controls (skipped), C has 1 control (skipped).  Only A
    # makes the cut → no pairs.
    assert anchor.studies == ["A"]
    assert anchor.coral_pairs.empty
    assert anchor.mmd_pairs.empty
