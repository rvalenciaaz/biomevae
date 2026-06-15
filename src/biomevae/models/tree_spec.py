"""Taxonomy tree spec data structures shared by PhILR-style models.

These utilities are independent of any particular model and provide:

* :class:`TreeSpec` — a serialisable description of a rooted taxonomy tree
  (parent pointers, leaf nodes, edges, depths, post-order traversal).
* :func:`build_tree_spec` — construct a :class:`TreeSpec` from a list of
  feature clade names and a taxonomy table on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from biomevae.taxonomy import TAX_LEVELS_ALL, load_taxonomy_table


RANK_LENGTHS = {
    "k": 2.0,
    "p": 1.75,
    "c": 1.5,
    "o": 1.25,
    "f": 1.0,
    "g": 0.75,
    "s": 0.5,
    "feature": 0.5,
}


@dataclass
class TreeSpec:
    parent: np.ndarray
    node_labels: List[str]
    leaf_nodes: np.ndarray
    edge_parent: np.ndarray
    edge_child: np.ndarray
    edge_length: np.ndarray
    edge_depth: np.ndarray
    node_depth: np.ndarray
    postorder: np.ndarray

    def to_json(self) -> Dict[str, List]:
        return {
            "parent": self.parent.tolist(),
            "node_labels": list(self.node_labels),
            "leaf_nodes": self.leaf_nodes.tolist(),
            "edge_parent": self.edge_parent.tolist(),
            "edge_child": self.edge_child.tolist(),
            "edge_length": self.edge_length.tolist(),
            "edge_depth": self.edge_depth.tolist(),
            "node_depth": self.node_depth.tolist(),
            "postorder": self.postorder.tolist(),
        }

    @staticmethod
    def from_json(payload: Dict[str, List]) -> "TreeSpec":
        return TreeSpec(
            parent=np.asarray(payload["parent"], dtype=np.int64),
            node_labels=list(payload["node_labels"]),
            leaf_nodes=np.asarray(payload["leaf_nodes"], dtype=np.int64),
            edge_parent=np.asarray(payload["edge_parent"], dtype=np.int64),
            edge_child=np.asarray(payload["edge_child"], dtype=np.int64),
            edge_length=np.asarray(payload["edge_length"], dtype=np.float32),
            edge_depth=np.asarray(payload["edge_depth"], dtype=np.float32),
            node_depth=np.asarray(payload["node_depth"], dtype=np.int64),
            postorder=np.asarray(payload["postorder"], dtype=np.int64),
        )


def build_tree_spec(
    feature_clades: List[str],
    taxonomy_path: str,
    branchlen_mode: str = "unit",
) -> TreeSpec:
    if branchlen_mode not in {"unit", "rank"}:
        raise ValueError("branchlen_mode must be 'unit' or 'rank'.")
    if not feature_clades:
        raise ValueError("feature_clades must contain at least one entry.")

    tax = load_taxonomy_table(taxonomy_path)
    tax_aligned = tax.reindex(feature_clades)
    if tax_aligned.isna().any().any():
        tax_aligned = tax_aligned.fillna({lvl: f"NA_{lvl}" for lvl in TAX_LEVELS_ALL})

    nodes: List[str] = ["root"]
    parent: List[int] = [-1]
    node_index: Dict[str, int] = {"root": 0}

    def get_idx(label: str) -> int:
        if label not in node_index:
            node_index[label] = len(nodes)
            nodes.append(label)
            parent.append(-1)
        return node_index[label]

    leaf_nodes: List[int] = []

    for clade in feature_clades:
        prev_idx = 0
        lineage = tax_aligned.loc[clade]
        for lvl in TAX_LEVELS_ALL:
            label = f"{lvl}::{lineage[lvl]}"
            idx = get_idx(label)
            if parent[idx] == -1:
                parent[idx] = prev_idx
            prev_idx = idx
        leaf_label = f"feature::{clade}"
        leaf_idx = get_idx(leaf_label)
        if parent[leaf_idx] == -1:
            parent[leaf_idx] = prev_idx
        leaf_nodes.append(leaf_idx)

    parent_arr = np.asarray(parent, dtype=np.int64)
    node_depth = np.zeros(len(nodes), dtype=np.int64)
    for idx in range(1, len(nodes)):
        node_depth[idx] = node_depth[parent_arr[idx]] + 1

    children: List[List[int]] = [[] for _ in range(len(nodes))]
    for idx in range(1, len(nodes)):
        children[parent_arr[idx]].append(idx)

    postorder: List[int] = []

    def _visit(node: int) -> None:
        for child in children[node]:
            _visit(child)
        postorder.append(node)

    _visit(0)
    postorder = [n for n in postorder if n != 0]

    edge_child = np.arange(1, len(nodes), dtype=np.int64)
    edge_parent = parent_arr[1:]

    edge_length = np.ones(len(edge_child), dtype=np.float32)
    if branchlen_mode == "rank":
        for i, child in enumerate(edge_child):
            label = nodes[child]
            level = label.split("::", 1)[0]
            edge_length[i] = float(RANK_LENGTHS.get(level, 1.0))

    edge_depth = node_depth[edge_child].astype(np.float32)

    return TreeSpec(
        parent=parent_arr,
        node_labels=nodes,
        leaf_nodes=np.asarray(leaf_nodes, dtype=np.int64),
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_length=edge_length,
        edge_depth=edge_depth,
        node_depth=node_depth,
        postorder=np.asarray(postorder, dtype=np.int64),
    )
