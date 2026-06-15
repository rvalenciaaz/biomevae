"""Legacy TreeSpec-based PhILR transform retained for :class:`biomevae.models.dsvae.DSVAE`.

The mainline PhILR family has migrated to :mod:`biomevae.models.philrvae` which
builds its ILR basis from a :class:`biomevae.models.taxonomy_tree.TaxonomyGraph`.
DSVAE still uses :class:`biomevae.models.tree_spec.TreeSpec`, so its PhILR
transform lives here to keep the two contracts cleanly separated.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.tree_spec import TreeSpec


def _build_philr_contrast_matrix(tree_spec: TreeSpec) -> np.ndarray:
    """Build the PhILR contrast matrix Ψ of shape ``(n_leaves, n_leaves-1)``.

    Uses Sequential Binary Partition (SBP) at every multi-furcating node.
    """
    parent = tree_spec.parent
    leaf_set = set(tree_spec.leaf_nodes.tolist())
    n_nodes = len(parent)
    n_leaves = len(leaf_set)

    children: Dict[int, List[int]] = defaultdict(list)
    for node in range(n_nodes):
        p = int(parent[node])
        if p >= 0:
            children[p].append(node)

    leaf_sets: Dict[int, frozenset] = {}
    for node in tree_spec.postorder:
        node = int(node)
        if node in leaf_set:
            leaf_sets[node] = frozenset([node])
        else:
            s: set = set()
            for c in children[node]:
                s |= leaf_sets.get(c, frozenset())
            leaf_sets[node] = frozenset(s)
    if 0 not in leaf_sets:
        s = set()
        for c in children[0]:
            s |= leaf_sets.get(c, frozenset())
        leaf_sets[0] = frozenset(s)

    leaf_to_idx = {int(ln): i for i, ln in enumerate(tree_spec.leaf_nodes)}

    rows: List[np.ndarray] = []
    for node in range(n_nodes):
        ch = children.get(node, [])
        if len(ch) < 2:
            continue
        child_leaves = [leaf_sets.get(c, frozenset()) for c in ch]

        remaining: set = set()
        for ls in child_leaves:
            remaining |= ls

        for i in range(len(ch) - 1):
            group_a = child_leaves[i]
            remaining -= group_a
            group_b = frozenset(remaining)
            r, s = len(group_a), len(group_b)
            if r == 0 or s == 0:
                continue

            row = np.zeros(n_leaves, dtype=np.float64)
            pos = np.sqrt(s / (r * (r + s)))
            neg = -np.sqrt(r / (s * (r + s)))
            for leaf in group_a:
                row[leaf_to_idx[leaf]] = pos
            for leaf in group_b:
                row[leaf_to_idx[leaf]] = neg
            rows.append(row)

    if not rows:
        raise ValueError(
            "PhILR contrast matrix is empty — the taxonomy tree has no "
            "internal node with >= 2 children."
        )
    psi_t = np.stack(rows, axis=0)
    return psi_t.T.astype(np.float32)


class TreeSpecPhILRTransform(nn.Module):
    """TreeSpec-based PhILR transform retained for DSVAE.

    Mirrors the API of the legacy
    :class:`biomevae.models.philrvae.PhILRTransform` that took a
    ``TreeSpec`` and an additive pseudocount.
    """

    def __init__(self, tree_spec: TreeSpec, pseudocount: float = 0.5) -> None:
        super().__init__()
        psi = _build_philr_contrast_matrix(tree_spec)
        self.register_buffer("psi", torch.from_numpy(psi))
        self.pseudocount = float(pseudocount)
        self.n_features = psi.shape[0]
        self.n_coords = psi.shape[1]

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = x_raw.float() + self.pseudocount
        x = x / x.sum(dim=1, keepdim=True)
        return torch.log(x) @ self.psi

    def inverse(self, coords: torch.Tensor) -> torch.Tensor:
        log_p = coords @ self.psi.T
        return F.softmax(log_p, dim=1)
