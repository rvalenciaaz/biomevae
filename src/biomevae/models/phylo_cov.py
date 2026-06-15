"""Phylogenetic structure utilities for PhyloDIVA.

Three pieces of machinery shared by every PhyloDIVA wrapper:

1. ``build_internal_aggregator`` — build dense ``(I_k, L)`` aggregators
   that sum leaf abundances up to internal nodes at depth ``k``.  One
   matrix per requested depth; differentiable matmul on a tensor of leaf
   counts produces clade-level abundances at that depth.  Driven by the
   same post-order accumulation logic as
   :func:`biomevae.models.taxonomy_tree.build_internal_sums_vector`, but in
   matrix form so the result lives on-GPU.

2. ``build_edge_parent_edge_index`` — precompute, for every edge in a
   ``TreeTopology``, the index of the *parent* edge (the edge whose
   child node equals the current edge's parent node), with ``-1`` for
   edges incident to the root.  Used by the BM-smoothness penalty on
   ``TreeSoftmaxDecoder.edge_logits``.

3. ``bm_edge_smoothness`` — Brownian-motion / Felsenstein-1985-style
   penalty on a per-edge tensor: ``mean over (e, parent(e)) of
   ((logit_e - logit_parent(e))^2 / edge_length_e)``.  Encourages
   neighbouring edges to carry similar values, which mimics the BM prior
   on a Gaussian process indexed by tree position.

For PhILR-style backbones the same idea is implemented in
``build_internal_node_parent_idx`` + ``bm_coord_smoothness`` on the
ILR-coordinate tensor (one balance per internal node).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from biomevae.models.taxonomy_tree import TaxonomyGraph
from biomevae.models.tree_spec import TreeSpec


__all__ = [
    "build_internal_aggregator",
    "build_edge_parent_edge_index",
    "build_internal_node_parent_idx",
    "bm_edge_smoothness",
    "bm_coord_smoothness",
]


# ---------------------------------------------------------------------------
# Hierarchical aggregators (TaxonomyGraph -> dense (I_k, L) matrices)
# ---------------------------------------------------------------------------


def build_internal_aggregator(
    taxg: TaxonomyGraph,
    leaf_node_ids: Sequence[int],
    depths: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Build ``(I_k, L)`` aggregators for each requested depth.

    Each aggregator ``A_k`` has rows indexed by internal nodes whose
    ``node_rank == k`` (rank=1 → kingdom, 2 → phylum, …; root has
    rank 0) and columns indexed by leaves in the order given.  Entry
    ``A_k[i, l]`` is 1 if leaf ``l`` descends from internal node ``i``,
    else 0.

    For an aggregator ``A_k`` and a leaf-abundance row vector
    ``x_leaf ∈ R^L``, ``A_k @ x_leaf`` gives the sum of all leaf counts
    underneath each depth-``k`` internal node — exactly what
    :func:`build_internal_sums_vector` produces for that depth.

    Empty depths (no internal nodes at that rank) are silently dropped.
    """
    parent = taxg.parent_of
    node_rank = taxg.node_rank.cpu().numpy().astype(np.int64)
    node_type = taxg.node_type.cpu().numpy().astype(np.int64)
    n_leaves = len(leaf_node_ids)

    # Walk from each leaf up to the root, recording the ancestor at every
    # depth on the way.  Costs O(L * tree_depth) — negligible.
    leaf_ancestors_at_depth: Dict[int, Dict[int, List[int]]] = {}
    for li, lid in enumerate(leaf_node_ids):
        cur = int(lid)
        # ascend to root
        while True:
            r = int(node_rank[cur])
            leaf_ancestors_at_depth.setdefault(r, {}).setdefault(cur, []).append(li)
            if cur not in parent:
                break
            cur = int(parent[cur])

    out: Dict[int, np.ndarray] = {}
    for d in depths:
        if d not in leaf_ancestors_at_depth:
            continue
        # Only keep internal-typed nodes (skip the case where a leaf
        # itself sits at the requested depth — at the leaf rank we'd just
        # recover the identity, which is what the leaf input already is).
        node_to_leaves = leaf_ancestors_at_depth[d]
        internal_nodes = sorted(
            n for n in node_to_leaves if int(node_type[n]) == 0
        )
        if not internal_nodes:
            continue
        A = np.zeros((len(internal_nodes), n_leaves), dtype=np.float32)
        for i, n in enumerate(internal_nodes):
            for li in node_to_leaves[n]:
                A[i, li] = 1.0
        out[d] = A
    return out


# ---------------------------------------------------------------------------
# Edge-parent-edge index (TreeTopology) and BM smoothness on edge tensors
# ---------------------------------------------------------------------------


def build_edge_parent_edge_index(
    edge_parent: np.ndarray, edge_child: np.ndarray,
) -> np.ndarray:
    """For each edge ``e``, return the index of the edge whose child is
    ``edge_parent[e]``, or ``-1`` if no such edge exists (root-incident).

    Both arrays are length ``n_edges``; ``edge_child`` values are unique
    (each non-root node has exactly one incoming edge), so the mapping
    ``child_node -> edge_index`` is well-defined.
    """
    n_edges = int(edge_parent.shape[0])
    child_to_edge: Dict[int, int] = {
        int(edge_child[e]): e for e in range(n_edges)
    }
    out = np.full((n_edges,), -1, dtype=np.int64)
    for e in range(n_edges):
        p = int(edge_parent[e])
        if p in child_to_edge:
            out[e] = child_to_edge[p]
    return out


def bm_edge_smoothness(
    edge_logits: torch.Tensor,
    parent_edge_idx: torch.Tensor,
    edge_length: torch.Tensor | None = None,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Brownian-motion smoothness penalty on a per-edge logit tensor.

    Parameters
    ----------
    edge_logits:
        ``(B, E)`` tensor (per-sample edge logits) or ``(E, K)`` weight
        tensor.  Penalty is computed along the last *non-edge* axis.
    parent_edge_idx:
        ``(E,)`` int64 with the index of each edge's parent edge, or
        ``-1`` for root-incident edges (which are masked out).
    edge_length:
        ``(E,)`` float tensor of branch lengths.  ``None`` falls back to
        unit lengths.

    Returns a scalar — the mean of
    ``(x[..., e] - x[..., parent(e)]) ** 2 / (edge_length[e] + eps)``
    over the batch (or first axis) and the masked edges.
    """
    if edge_logits.dim() < 2:
        raise ValueError("edge_logits must have at least 2 dims (..., E).")
    if parent_edge_idx.shape[0] != edge_logits.shape[-1]:
        raise ValueError(
            f"parent_edge_idx length ({parent_edge_idx.shape[0]}) "
            f"does not match edge axis ({edge_logits.shape[-1]})."
        )
    mask = parent_edge_idx >= 0
    if not bool(mask.any()):
        return edge_logits.new_zeros(())
    valid = mask.nonzero(as_tuple=False).squeeze(-1).long()
    parent_idx = parent_edge_idx[valid].long()
    child_vals = edge_logits.index_select(-1, valid)
    parent_vals = edge_logits.index_select(-1, parent_idx)
    diff_sq = (child_vals - parent_vals).pow(2)
    if edge_length is not None:
        bl = edge_length[valid].clamp_min(eps).to(diff_sq.dtype)
        diff_sq = diff_sq / bl
    return diff_sq.mean()


# ---------------------------------------------------------------------------
# PhILR-side: per-internal-node parent index and BM smoothness on coords
# ---------------------------------------------------------------------------


def build_internal_node_parent_idx(
    tree_spec: TreeSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute, for each internal node of ``tree_spec``, the index of
    its parent internal node *in the n_coords ordering*.

    PhILR balances are indexed by internal nodes (one balance per
    internal node, ``n_coords = p - 1``).  The order is determined by
    :func:`biomevae.models.philrvae._build_philr_contrast_matrix`, which
    walks ``tree_spec.postorder`` and assigns one column per visited
    internal node.

    Returns
    -------
    internal_node_ids : (n_coords,) int64
        The internal node id for each PhILR coordinate, in coord order.
    parent_coord_idx : (n_coords,) int64
        For each coord ``c``, the coord index of its parent internal
        node, or ``-1`` if the parent is the root (which carries no
        balance).
    edge_length : (n_coords,) float32
        Branch length of the edge from each internal node to its parent
        (taken from ``tree_spec.edge_length``).
    """
    parent = tree_spec.parent
    leaf_set = set(tree_spec.leaf_nodes.tolist())
    postorder = list(tree_spec.postorder.tolist())
    edge_lengths = tree_spec.edge_length

    # Mirror philrvae._build_philr_contrast_matrix: visit postorder, and
    # for every *internal* node that has children, it contributes one
    # balance.  Matches that the PhILR contrast matrix has exactly p-1
    # columns, one per internal node in postorder visit order.
    internal_node_ids: List[int] = []
    for node in postorder:
        node = int(node)
        if node in leaf_set:
            continue
        # Skip nodes with no children (degenerate); a real internal node
        # in this taxonomy always has >=1 child.
        # We retain only nodes that *are* parents in the tree.
        if not (parent == node).any():
            continue
        internal_node_ids.append(node)

    n_coords = len(internal_node_ids)
    node_to_coord: Dict[int, int] = {n: i for i, n in enumerate(internal_node_ids)}

    parent_coord_idx = np.full((n_coords,), -1, dtype=np.int64)
    coord_edge_length = np.zeros((n_coords,), dtype=np.float32)
    for c, n in enumerate(internal_node_ids):
        p = int(parent[n])
        if p >= 0 and p in node_to_coord:
            parent_coord_idx[c] = node_to_coord[p]
        # Edge length for the edge n -> p; fall back to 1.0 if not found.
        # tree_spec.edge_child / edge_parent are indexed in build order;
        # find the edge whose child is ``n``.
        match = np.where(tree_spec.edge_child == n)[0]
        if match.size:
            coord_edge_length[c] = float(edge_lengths[int(match[0])])
        else:
            coord_edge_length[c] = 1.0

    return (
        np.asarray(internal_node_ids, dtype=np.int64),
        parent_coord_idx,
        coord_edge_length,
    )


def bm_coord_smoothness(
    coords: torch.Tensor,
    parent_coord_idx: torch.Tensor,
    edge_length: torch.Tensor | None = None,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    """BM smoothness penalty on a per-coord tensor (e.g. PhILR coords).

    ``coords`` is shape ``(B, n_coords)``; the penalty masks out coords
    whose parent is the tree root (``parent_coord_idx == -1``).  Identical
    semantics to :func:`bm_edge_smoothness`.
    """
    return bm_edge_smoothness(coords, parent_coord_idx, edge_length, eps=eps)
