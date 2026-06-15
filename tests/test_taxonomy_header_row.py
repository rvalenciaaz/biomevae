"""Regression tests for stray-header handling in taxonomy tree construction.

When a ``phyla.tsv`` exported with a MetaPhlAn-style ``clade_name`` header is
read positionally (``has_header=False``, the default used across the training,
embedding, and interpret CLIs), the header line must not be parsed as a taxon.
Otherwise it injects a spurious leaf (literally ``clade_name``) that has no
matching abundance-table feature and makes strict leaf/table alignment fail with
``missing=[clade_name]``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from biomevae.models.taxonomy_tree import (
    align_table_to_tree_leaves,
    build_taxonomy_graph_from_phyla_tsv,
)

_ROWS = [
    ["t__SGB1", "k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__O", "f__F", "g__G", "s__S1"],
    ["t__SGB2", "k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__O", "f__F", "g__G", "s__S2"],
    ["t__SGB3", "k__Bacteria", "p__Bacteroidetes", "c__Bact", "o__O2", "f__F2", "g__G2", "s__S3"],
]
_HEADER = ["clade_name", "kingdom", "phylum", "class", "order", "family", "genus", "species"]


def _write(path: Path, rows) -> Path:
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return path


def _leaves(taxg):
    return sorted(taxg.node_names[i] for i in taxg.leaf_ids)


def test_clade_name_header_dropped_when_read_positionally(tmp_path):
    phyla = _write(tmp_path / "phyla.tsv", [_HEADER, *_ROWS])
    taxg = build_taxonomy_graph_from_phyla_tsv(
        phyla, has_header=False, on_duplicate_leaf="ignore_same"
    )
    leaves = _leaves(taxg)
    assert "clade_name" not in leaves
    assert leaves == ["t__SGB1", "t__SGB2", "t__SGB3"]


def test_strict_alignment_succeeds_after_header_drop(tmp_path):
    phyla = _write(tmp_path / "phyla.tsv", [_HEADER, *_ROWS])
    taxg = build_taxonomy_graph_from_phyla_tsv(
        phyla, has_header=False, on_duplicate_leaf="ignore_same"
    )
    sgb = (
        pd.DataFrame(
            {"clade_name": ["t__SGB1", "t__SGB2", "t__SGB3"], "s1": [1, 2, 3], "s2": [0, 5, 1]}
        )
        .set_index("clade_name")
        .T
    )
    _X, _names, report = align_table_to_tree_leaves(sgb, taxg, strict=True)
    assert report.missing_tree_leaves == []
    assert report.n_tree_leaves == report.n_table_features == 3


def test_headerless_table_is_unchanged(tmp_path):
    phyla = _write(tmp_path / "phyla.tsv", _ROWS)
    taxg = build_taxonomy_graph_from_phyla_tsv(
        phyla, has_header=False, on_duplicate_leaf="ignore_same"
    )
    assert _leaves(taxg) == ["t__SGB1", "t__SGB2", "t__SGB3"]


def test_real_taxon_resembling_header_is_not_dropped(tmp_path):
    # A genuine leaf whose name merely contains "clade_name" must survive.
    rows = [["t__clade_nameX", "k__Bacteria", "p__F"], ["t__SGB2", "k__Bacteria", "p__F"]]
    phyla = _write(tmp_path / "phyla.tsv", rows)
    taxg = build_taxonomy_graph_from_phyla_tsv(
        phyla, has_header=False, on_duplicate_leaf="ignore_same"
    )
    assert _leaves(taxg) == ["t__SGB2", "t__clade_nameX"]


def test_has_header_true_path_unaffected(tmp_path):
    phyla = _write(tmp_path / "phyla.tsv", [_HEADER, *_ROWS])
    taxg = build_taxonomy_graph_from_phyla_tsv(
        phyla, has_header=True, on_duplicate_leaf="ignore_same"
    )
    assert _leaves(taxg) == ["t__SGB1", "t__SGB2", "t__SGB3"]
