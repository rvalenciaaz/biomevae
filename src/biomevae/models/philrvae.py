"""PhILR-VAE compatible with taxonomy_tree.py utilities.


Recommended usage
-----------------
1. Build aligned data with ``build_philrvae_dataset``.
2. Construct the model with the returned ``taxg``.
3. Use:
   * likelihood="philr_gaussian" for relative-abundance/compositional data.
   * likelihood="multinomial" for true integer counts.
   * likelihood="dirichlet_multinomial" for overdispersed true counts.
   * likelihood="dirichlet_tree_multinomial" for tree-local overdispersed true counts.
   * likelihood="dirichlet_tree" for continuous relative-abundance compositions.

The model deliberately does not use an independent Negative Binomial likelihood
over leaves, because that is usually inappropriate for closed microbiome
relative-abundance profiles and suboptimal for fixed-depth count profiles.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# taxonomy_tree.py compatibility
# ---------------------------------------------------------------------------

from biomevae.models import taxonomy_tree as _tax_utils


TaxonomyGraph = _tax_utils.TaxonomyGraph
build_taxonomy_graph_from_phyla_tsv = _tax_utils.build_taxonomy_graph_from_phyla_tsv


_RANK_PREFIX_RE = re.compile(r"^[a-zA-Z]__")


def _strip_rank_prefix(x: object) -> str:
    s = "" if x is None else str(x).strip()
    if _RANK_PREFIX_RE.match(s):
        return s.split("__", 1)[1].strip()
    return s


AlignmentReport = _tax_utils.AlignmentReport
load_feature_table_as_samples_by_feature = _tax_utils.load_feature_table_as_samples_by_feature
align_table_to_tree_leaves = _tax_utils.align_table_to_tree_leaves
close_composition = _tax_utils.close_composition
validate_nonnegative_integer_counts = _tax_utils.validate_nonnegative_integer_counts
aggregate_leaf_matrix_to_nodes = _tax_utils.aggregate_leaf_matrix_to_nodes


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DataKind = Literal["counts", "relative"]

LikelihoodName = Literal[
    "philr_gaussian",
    "multinomial",
    "dirichlet_multinomial",
    "dirichlet_tree_multinomial",
    "dirichlet_tree",
]


# ---------------------------------------------------------------------------
# TaxonomyGraph tree extraction and PhILR basis
# ---------------------------------------------------------------------------

@dataclass
class PhILRTreeBasis:
    """Tree-derived arrays needed by the PhILR-VAE."""

    psi: np.ndarray                         # (n_leaves, n_leaves - 1)
    root: int
    parent: np.ndarray                      # (n_nodes,), root has -1
    children: List[List[int]]
    postorder: np.ndarray                   # leaves before parents
    node_depth: np.ndarray
    leaf_nodes: np.ndarray                  # node IDs in feature order

    group_parent: np.ndarray
    group_child_nodes: np.ndarray
    group_child_mask: np.ndarray
    group_depth: np.ndarray

    contrast_node: np.ndarray
    contrast_pos_size: np.ndarray
    contrast_neg_size: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.parent.size)

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_nodes.size)

    @property
    def n_coords(self) -> int:
        return int(self.psi.shape[1])

    @property
    def n_groups(self) -> int:
        return int(self.group_parent.size)


def _extract_rooted_tree_from_taxg(
    taxg: TaxonomyGraph,
) -> Tuple[int, np.ndarray, List[List[int]], np.ndarray, np.ndarray]:
    """Infer root, parent array, children list, postorder and node depths."""
    n_nodes = len(taxg.node_names)

    parent = np.full(n_nodes, -1, dtype=np.int64)
    for child, par in taxg.parent_of.items():
        child_i = int(child)
        par_i = int(par)
        if child_i < 0 or child_i >= n_nodes:
            raise ValueError(f"parent_of contains invalid child node {child_i}.")
        if par_i < 0 or par_i >= n_nodes:
            raise ValueError(f"parent_of contains invalid parent node {par_i}.")
        parent[child_i] = par_i

    roots = np.flatnonzero(parent < 0)
    if roots.size != 1:
        raise ValueError(f"Expected exactly one root in TaxonomyGraph; found {roots.tolist()}.")

    root = int(roots[0])

    children: List[List[int]] = [[] for _ in range(n_nodes)]
    for par, ch in taxg.children_of.items():
        par_i = int(par)
        if par_i < 0 or par_i >= n_nodes:
            raise ValueError(f"children_of contains invalid parent node {par_i}.")
        children[par_i] = [int(c) for c in ch]

    postorder: List[int] = []
    node_depth = np.full(n_nodes, -1, dtype=np.int64)
    visiting = np.zeros(n_nodes, dtype=bool)

    def dfs(node: int, depth: int) -> None:
        if visiting[node]:
            raise ValueError("Cycle detected in TaxonomyGraph.")
        if node_depth[node] >= 0:
            return

        visiting[node] = True
        node_depth[node] = depth

        for child in children[node]:
            if parent[child] != node:
                raise ValueError(
                    f"Inconsistent TaxonomyGraph: node {child} is listed as child "
                    f"of {node}, but parent_of gives {parent[child]}."
                )
            dfs(child, depth + 1)

        visiting[node] = False
        postorder.append(node)

    dfs(root, 0)

    if len(postorder) != n_nodes or np.any(node_depth < 0):
        missing = np.flatnonzero(node_depth < 0).tolist()
        raise ValueError(f"TaxonomyGraph is disconnected from root {root}; missing nodes={missing}.")

    return root, parent, children, np.asarray(postorder, dtype=np.int64), node_depth


def build_philr_basis_from_taxonomy_graph(
    taxg: TaxonomyGraph,
    *,
    sort_children: bool = True,
    check_orthonormal: bool = True,
    atol: float = 1e-5,
) -> PhILRTreeBasis:
    """Build an unweighted PhILR/ILR basis from ``TaxonomyGraph``.

    For a multifurcating node with children c1..ck, the script uses a
    deterministic sequential binary partition:

        c1 vs c2..ck,
        c2 vs c3..ck,
        ...
        c{k-1} vs ck.
    """
    root, parent, children, postorder, node_depth = _extract_rooted_tree_from_taxg(taxg)

    leaf_nodes = np.asarray([int(x) for x in taxg.leaf_ids], dtype=np.int64)
    if leaf_nodes.size < 2:
        raise ValueError("PhILR requires at least two leaves.")

    n_nodes = len(taxg.node_names)
    leaf_set = set(int(x) for x in leaf_nodes.tolist())

    if len(leaf_set) != leaf_nodes.size:
        raise ValueError("TaxonomyGraph.leaf_ids contains duplicate leaves.")

    for leaf in leaf_nodes.tolist():
        if leaf < 0 or leaf >= n_nodes:
            raise ValueError(f"Leaf node {leaf} is outside 0..{n_nodes - 1}.")
        if children[leaf]:
            raise ValueError(f"Leaf node {leaf} has children; leaves must be terminal.")

    terminal_nodes = {i for i, ch in enumerate(children) if len(ch) == 0}
    if terminal_nodes != leaf_set:
        missing = sorted(terminal_nodes - leaf_set)
        extra = sorted(leaf_set - terminal_nodes)
        raise ValueError(
            "TaxonomyGraph.leaf_ids must match terminal nodes. "
            f"Unlisted terminals={missing}; listed non-terminals={extra}."
        )

    leaf_to_idx = {int(leaf): i for i, leaf in enumerate(leaf_nodes.tolist())}

    leaf_sets: Dict[int, frozenset[int]] = {}
    for node in postorder.tolist():
        node_i = int(node)
        if node_i in leaf_set:
            leaf_sets[node_i] = frozenset([node_i])
        else:
            s: set[int] = set()
            for child in children[node_i]:
                s.update(leaf_sets[int(child)])
            if not s:
                raise ValueError(f"Internal node {node_i} has no descendant leaves.")
            leaf_sets[node_i] = frozenset(s)

    def ordered_children(node: int) -> List[int]:
        ch = list(children[int(node)])
        if not sort_children:
            return ch
        return sorted(
            ch,
            key=lambda c: (
                min(leaf_to_idx[int(leaf)] for leaf in leaf_sets[int(c)]),
                int(c),
            ),
        )

    rows: List[np.ndarray] = []
    contrast_node: List[int] = []
    contrast_pos_size: List[int] = []
    contrast_neg_size: List[int] = []

    for node in postorder[::-1].tolist():
        node_i = int(node)
        ch = ordered_children(node_i)
        if len(ch) < 2:
            continue

        child_leaf_sets = [set(leaf_sets[int(c)]) for c in ch]

        remaining: set[int] = set()
        for ls in child_leaf_sets:
            remaining.update(ls)

        for i in range(len(ch) - 1):
            group_a = set(child_leaf_sets[i])
            remaining.difference_update(group_a)
            group_b = set(remaining)

            r = len(group_a)
            s = len(group_b)
            if r == 0 or s == 0:
                raise ValueError(f"Empty SBP side at node {node_i}.")

            row = np.zeros(len(leaf_nodes), dtype=np.float64)

            pos = math.sqrt(s / (r * (r + s)))
            neg = -math.sqrt(r / (s * (r + s)))

            for leaf in group_a:
                row[leaf_to_idx[int(leaf)]] = pos
            for leaf in group_b:
                row[leaf_to_idx[int(leaf)]] = neg

            rows.append(row)
            contrast_node.append(node_i)
            contrast_pos_size.append(r)
            contrast_neg_size.append(s)

    if not rows:
        raise ValueError("Tree has no internal node with at least two children.")

    psi_t = np.stack(rows, axis=0)
    psi = psi_t.T.astype(np.float32)

    expected = len(leaf_nodes) - 1
    if psi.shape[1] != expected:
        raise ValueError(
            f"PhILR basis has {psi.shape[1]} contrasts but expected {expected}. "
            "Check for disconnected nodes, non-terminal leaves or malformed topology."
        )

    if check_orthonormal:
        gram = psi.T @ psi
        target = np.eye(expected, dtype=np.float32)
        zero_sum = psi.T @ np.ones((psi.shape[0],), dtype=np.float32)

        if not np.allclose(gram, target, atol=atol, rtol=0.0):
            err = float(np.max(np.abs(gram - target)))
            raise ValueError(f"PhILR basis is not orthonormal; max Gram error={err:.3g}.")

        if not np.allclose(zero_sum, 0.0, atol=atol, rtol=0.0):
            err = float(np.max(np.abs(zero_sum)))
            raise ValueError(f"PhILR basis columns are not zero-sum; max error={err:.3g}.")

    groups: List[Tuple[int, List[int]]] = []
    for node in range(n_nodes):
        ch = ordered_children(node)
        if len(ch) >= 2:
            groups.append((int(node), [int(c) for c in ch]))

    max_children = max((len(ch) for _, ch in groups), default=0)
    group_parent = np.asarray([p for p, _ in groups], dtype=np.int64)
    group_child_nodes = np.full((len(groups), max_children), -1, dtype=np.int64)
    group_child_mask = np.zeros((len(groups), max_children), dtype=bool)

    for g, (_parent, ch) in enumerate(groups):
        group_child_nodes[g, : len(ch)] = np.asarray(ch, dtype=np.int64)
        group_child_mask[g, : len(ch)] = True

    group_depth = (
        node_depth[group_parent].astype(np.int64)
        if len(group_parent)
        else np.zeros((0,), dtype=np.int64)
    )

    return PhILRTreeBasis(
        psi=psi,
        root=root,
        parent=parent,
        children=children,
        postorder=postorder,
        node_depth=node_depth,
        leaf_nodes=leaf_nodes,
        group_parent=group_parent,
        group_child_nodes=group_child_nodes,
        group_child_mask=group_child_mask,
        group_depth=group_depth,
        contrast_node=np.asarray(contrast_node, dtype=np.int64),
        contrast_pos_size=np.asarray(contrast_pos_size, dtype=np.int64),
        contrast_neg_size=np.asarray(contrast_neg_size, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_philrvae_dataset(
    sgb_table_tsv: Union[str, Path],
    phyla_tsv: Union[str, Path],
    *,
    data_kind: DataKind = "relative",
    keep_prefixes: bool = False,
    strict_alignment: bool = True,
    allow_missing_leaves: bool = False,
    allow_extra_features: bool = True,
    min_matched_fraction: float = 0.95,
    taxonomy_has_header: bool = False,
) -> Tuple[TaxonomyGraph, torch.Tensor, torch.Tensor, List[str], List[str], AlignmentReport]:
    """Build an aligned dataset for ``PhILRVAE``.

    Returns
    -------
    taxg, X_leaf, X_nodes, sample_ids, leaf_names, alignment_report
    """
    phyla_tsv = Path(phyla_tsv)
    sgb_table_tsv = Path(sgb_table_tsv)

    try:
        taxg = build_taxonomy_graph_from_phyla_tsv(
            phyla_tsv,
            keep_prefixes=keep_prefixes,
            has_header=taxonomy_has_header,
            fill_missing_intermediate=True,
            on_duplicate_leaf="ignore_same",
        )
    except TypeError:
        taxg = build_taxonomy_graph_from_phyla_tsv(
            phyla_tsv,
            keep_prefixes=keep_prefixes,
        )

    Xdf, sample_ids, _feature_ids = load_feature_table_as_samples_by_feature(sgb_table_tsv)

    X_leaf_np, leaf_names, report = align_table_to_tree_leaves(
        Xdf,
        taxg,
        strict=strict_alignment,
        allow_missing_leaves=allow_missing_leaves,
        allow_extra_features=allow_extra_features,
        min_matched_fraction=min_matched_fraction,
    )

    if data_kind == "counts":
        X_leaf_np = validate_nonnegative_integer_counts(X_leaf_np)
    elif data_kind == "relative":
        X_leaf_np = close_composition(X_leaf_np)
    else:
        raise ValueError("data_kind must be 'counts' or 'relative'.")

    X_nodes_np = aggregate_leaf_matrix_to_nodes(taxg, X_leaf_np)

    return (
        taxg,
        torch.from_numpy(X_leaf_np.astype(np.float32)),
        torch.from_numpy(X_nodes_np.astype(np.float32)),
        sample_ids,
        leaf_names,
        report,
    )


# ---------------------------------------------------------------------------
# PhILR transform
# ---------------------------------------------------------------------------

class PhILRTransform(nn.Module):
    """Differentiable unweighted PhILR forward/inverse transform."""

    def __init__(
        self,
        basis: PhILRTreeBasis,
        *,
        count_pseudocount: float = 0.5,
        relative_pseudocount: float = 1e-6,
    ) -> None:
        super().__init__()

        if count_pseudocount < 0:
            raise ValueError("count_pseudocount must be non-negative.")
        if relative_pseudocount < 0:
            raise ValueError("relative_pseudocount must be non-negative.")

        self.register_buffer("psi", torch.as_tensor(basis.psi, dtype=torch.float32))

        self.count_pseudocount = float(count_pseudocount)
        self.relative_pseudocount = float(relative_pseudocount)
        self.n_features = int(basis.n_leaves)
        self.n_coords = int(basis.n_coords)

    @staticmethod
    def _close(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        x = x.clamp_min(0.0)
        total = x.sum(dim=1, keepdim=True)
        return x / total.clamp_min(eps)

    def observation_to_composition(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> torch.Tensor:
        x = x.float()

        if x.ndim != 2 or x.size(1) != self.n_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.n_features}); got {tuple(x.shape)}."
            )

        if not torch.isfinite(x).all() or (x < 0).any():
            raise ValueError("Input contains negative or non-finite values.")

        if data_kind == "counts":
            p = x + self.count_pseudocount
            return self._close(p)

        if data_kind == "relative":
            p = self._close(x)
            if self.relative_pseudocount > 0:
                p = p + self.relative_pseudocount
                p = self._close(p)
            return p

        raise ValueError("data_kind must be 'counts' or 'relative'.")

    def forward(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> torch.Tensor:
        p = self.observation_to_composition(x, data_kind=data_kind)
        return torch.log(p.clamp_min(1e-30)) @ self.psi

    def inverse(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.size(1) != self.n_coords:
            raise ValueError(
                f"Expected coords with shape (batch, {self.n_coords}); "
                f"got {tuple(coords.shape)}."
            )

        clr = coords @ self.psi.T
        return F.softmax(clr, dim=1)


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        *,
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev = int(in_dim)

        for h in hidden:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(float(dropout)))
            prev = h

        if out_dim is not None:
            layers.append(nn.Linear(prev, int(out_dim)))
            prev = int(out_dim)

        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _inverse_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("x must be positive.")
    if x > 20:
        return x
    return math.log(math.expm1(x))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PhILRVAE(nn.Module):
    """TaxonomyGraph-compatible PhILR-VAE.

    Input
    -----
    ``forward`` and ``loss`` expect samples x leaves tensors in the exact order
    returned by ``taxg.leaf_ids``. Use ``build_philrvae_dataset`` to create this.
    """

    VALID_LIKELIHOODS = {
        "philr_gaussian",
        "multinomial",
        "dirichlet_multinomial",
        "dirichlet_tree_multinomial",
        "dirichlet_tree",
    }

    def __init__(
        self,
        taxg: TaxonomyGraph,
        latent_dim: int,
        *,
        n_features: Optional[int] = None,
        hidden: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
        count_pseudocount: float = 0.5,
        relative_pseudocount: float = 1e-6,
        default_likelihood: LikelihoodName = "philr_gaussian",
        init_coord_scale: float = 0.5,
        init_concentration: float = 50.0,
        min_coord_scale: float = 1e-4,
        min_concentration: float = 1e-3,
        sort_children: bool = True,
        check_basis: bool = True,
    ) -> None:
        super().__init__()

        if default_likelihood not in self.VALID_LIKELIHOODS:
            raise ValueError(f"Unknown default_likelihood={default_likelihood!r}.")

        if hidden is None:
            hidden = (256, 128)

        self.taxg = taxg
        self.basis = build_philr_basis_from_taxonomy_graph(
            taxg,
            sort_children=sort_children,
            check_orthonormal=check_basis,
        )

        if n_features is not None and int(n_features) != self.basis.n_leaves:
            raise ValueError(
                f"n_features={n_features} does not match number of taxonomy leaves "
                f"({self.basis.n_leaves})."
            )

        self.default_likelihood: LikelihoodName = default_likelihood
        self.latent_dim = int(latent_dim)
        self.n_features = int(self.basis.n_leaves)
        self.n_coords = int(self.basis.n_coords)
        self.n_nodes = int(self.basis.n_nodes)
        self.min_coord_scale = float(min_coord_scale)
        self.min_concentration = float(min_concentration)

        self.philr = PhILRTransform(
            self.basis,
            count_pseudocount=count_pseudocount,
            relative_pseudocount=relative_pseudocount,
        )

        hidden = tuple(int(h) for h in hidden)

        self.encoder_trunk = _MLP(
            self.n_coords,
            hidden,
            out_dim=None,
            dropout=dropout,
            layer_norm=True,
        )
        self.fc_mu = nn.Linear(self.encoder_trunk.out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_trunk.out_dim, self.latent_dim)

        self.decoder = _MLP(
            self.latent_dim,
            tuple(reversed(hidden)),
            out_dim=self.n_coords,
            dropout=dropout,
            layer_norm=True,
        )

        nn.init.constant_(self.fc_logvar.bias, -2.0)

        self.raw_coord_scale = nn.Parameter(
            torch.full(
                (self.n_coords,),
                _inverse_softplus(float(init_coord_scale)),
                dtype=torch.float32,
            )
        )

        raw_conc = _inverse_softplus(float(init_concentration))
        self.raw_flat_concentration = nn.Parameter(torch.tensor(raw_conc, dtype=torch.float32))

        max_depth = int(np.max(self.basis.group_depth)) if self.basis.n_groups else 0
        self.raw_depth_concentration = nn.Parameter(
            torch.full((max_depth + 1,), raw_conc, dtype=torch.float32)
        )
        self.group_concentration_delta = nn.Parameter(torch.zeros(self.basis.n_groups))

        self.register_buffer("leaf_nodes", torch.as_tensor(self.basis.leaf_nodes, dtype=torch.long))
        self.register_buffer("group_parent", torch.as_tensor(self.basis.group_parent, dtype=torch.long))
        self.register_buffer(
            "group_child_nodes",
            torch.as_tensor(self.basis.group_child_nodes, dtype=torch.long),
        )
        self.register_buffer(
            "group_child_mask",
            torch.as_tensor(self.basis.group_child_mask, dtype=torch.bool),
        )
        self.register_buffer("group_depth", torch.as_tensor(self.basis.group_depth, dtype=torch.long))

        self._children: List[List[int]] = self.basis.children
        self._postorder: List[int] = [int(x) for x in self.basis.postorder.tolist()]

    def coord_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_coord_scale) + self.min_coord_scale

    def flat_concentration(self) -> torch.Tensor:
        return F.softplus(self.raw_flat_concentration) + self.min_concentration

    def group_concentration(self) -> torch.Tensor:
        if self.group_depth.numel() == 0:
            return self.raw_depth_concentration.new_zeros((0,))

        raw = (
            self.raw_depth_concentration.index_select(0, self.group_depth)
            + self.group_concentration_delta
        )
        return F.softplus(raw) + self.min_concentration

    def concentration_regularization(self) -> torch.Tensor:
        if self.group_concentration_delta.numel() == 0:
            return self.raw_depth_concentration.new_zeros(())
        return self.group_concentration_delta.pow(2).mean()

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-30.0, 20.0)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_per_sample(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        logvar = logvar.clamp(-30.0, 20.0)
        per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)

        if free_bits > 0:
            per_dim = per_dim.clamp_min(float(free_bits))

        return per_dim.sum(dim=-1)

    def encode_coords(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(coords)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def encode(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = self.philr(x, data_kind=data_kind)
        return self.encode_coords(coords)

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        coord_mu = self.decoder(z)
        leaf_prob = self.philr.inverse(coord_mu)
        return {
            "coord_mu": coord_mu,
            "leaf_prob": leaf_prob,
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> Dict[str, torch.Tensor]:
        obs_coords = self.philr(x, data_kind=data_kind)
        mu_z, logvar_z = self.encode_coords(obs_coords)
        z = self.reparam(mu_z, logvar_z)
        dec = self.decode(z)

        return {
            "obs_coords": obs_coords,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            **dec,
        }

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
        use_mean: bool = True,
    ) -> Dict[str, torch.Tensor]:
        obs_coords = self.philr(x, data_kind=data_kind)
        mu_z, logvar_z = self.encode_coords(obs_coords)
        z = mu_z if use_mean else self.reparam(mu_z, logvar_z)
        return self.decode(z)

    def aggregate_leaf_to_nodes(self, leaf_values: torch.Tensor) -> torch.Tensor:
        if leaf_values.ndim != 2 or leaf_values.size(1) != self.n_features:
            raise ValueError(
                f"Expected leaf_values with shape (batch, {self.n_features}); "
                f"got {tuple(leaf_values.shape)}."
            )

        node_values = leaf_values.new_zeros((leaf_values.size(0), self.n_nodes))
        node_values[:, self.leaf_nodes] = leaf_values

        for node in self._postorder:
            ch = self._children[node]
            if ch:
                node_values[:, node] = node_values[:, ch].sum(dim=1)

        return node_values

    @staticmethod
    def _validate_count_matrix(x: torch.Tensor, *, atol: float = 1e-4) -> None:
        if not torch.isfinite(x).all() or (x < 0).any():
            raise ValueError("Count matrix contains negative or non-finite values.")

        if not torch.allclose(x, x.round(), atol=atol, rtol=0.0):
            raise ValueError("Count likelihood requested, but x contains non-integer values.")

    def philr_gaussian_nll(
        self,
        obs_coords: torch.Tensor,
        coord_mu: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.coord_scale().unsqueeze(0)
        resid = (obs_coords - coord_mu) / scale

        return 0.5 * (
            resid.pow(2)
            + 2.0 * torch.log(scale)
            + math.log(2.0 * math.pi)
        ).sum(dim=1)

    def multinomial_nll(
        self,
        x_counts: torch.Tensor,
        leaf_prob: torch.Tensor,
        *,
        validate_counts: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if validate_counts:
            self._validate_count_matrix(x_counts)

        x = x_counts.float()
        n = x.sum(dim=1)
        logp = leaf_prob.clamp_min(eps).log()

        ll = (
            torch.lgamma(n + 1.0)
            - torch.lgamma(x + 1.0).sum(dim=1)
            + (x * logp).sum(dim=1)
        )
        return -ll

    def dirichlet_multinomial_nll(
        self,
        x_counts: torch.Tensor,
        leaf_prob: torch.Tensor,
        *,
        validate_counts: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if validate_counts:
            self._validate_count_matrix(x_counts)

        x = x_counts.float()
        n = x.sum(dim=1)

        concentration = self.flat_concentration()
        alpha = leaf_prob.clamp_min(eps) * concentration + eps
        alpha0 = alpha.sum(dim=1)

        ll = (
            torch.lgamma(n + 1.0)
            - torch.lgamma(x + 1.0).sum(dim=1)
            + torch.lgamma(alpha0)
            - torch.lgamma(n + alpha0)
            + (torch.lgamma(x + alpha) - torch.lgamma(alpha)).sum(dim=1)
        )
        return -ll

    def dirichlet_tree_multinomial_nll(
        self,
        x_counts: torch.Tensor,
        leaf_prob: torch.Tensor,
        *,
        validate_counts: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if validate_counts:
            self._validate_count_matrix(x_counts)

        x_nodes = self.aggregate_leaf_to_nodes(x_counts.float())
        p_nodes = self.aggregate_leaf_to_nodes(leaf_prob)
        concentrations = self.group_concentration()

        nll = x_counts.new_zeros(x_counts.size(0), dtype=torch.float32)

        for g in range(int(self.group_parent.numel())):
            mask = self.group_child_mask[g]
            child_nodes = self.group_child_nodes[g, mask]

            xg = x_nodes.index_select(1, child_nodes)
            ng = xg.sum(dim=1)

            pg = p_nodes.index_select(1, child_nodes)
            pg = pg / pg.sum(dim=1, keepdim=True).clamp_min(eps)

            alpha = pg.clamp_min(eps) * concentrations[g].clamp_min(eps) + eps
            alpha0 = alpha.sum(dim=1)

            ll = (
                torch.lgamma(ng + 1.0)
                - torch.lgamma(xg + 1.0).sum(dim=1)
                + torch.lgamma(alpha0)
                - torch.lgamma(ng + alpha0)
                + (torch.lgamma(xg + alpha) - torch.lgamma(alpha)).sum(dim=1)
            )
            nll = nll - ll

        return nll

    def dirichlet_tree_nll(
        self,
        x: torch.Tensor,
        leaf_prob: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
        observation_pseudocount: float = 1e-6,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        obs_leaf = self.philr.observation_to_composition(x, data_kind=data_kind)
        x_nodes = self.aggregate_leaf_to_nodes(obs_leaf)
        p_nodes = self.aggregate_leaf_to_nodes(leaf_prob)
        concentrations = self.group_concentration()

        nll = x.new_zeros(x.size(0), dtype=torch.float32)

        for g in range(int(self.group_parent.numel())):
            mask = self.group_child_mask[g]
            child_nodes = self.group_child_nodes[g, mask]
            k = int(child_nodes.numel())

            xg = x_nodes.index_select(1, child_nodes)
            total = xg.sum(dim=1, keepdim=True)

            obs_split = (xg + float(observation_pseudocount)) / (
                total + float(observation_pseudocount) * k
            ).clamp_min(eps)

            pg = p_nodes.index_select(1, child_nodes)
            pg = pg / pg.sum(dim=1, keepdim=True).clamp_min(eps)

            alpha = pg.clamp_min(eps) * concentrations[g].clamp_min(eps) + eps
            alpha0 = alpha.sum(dim=1)

            ll = (
                torch.lgamma(alpha0)
                - torch.lgamma(alpha).sum(dim=1)
                + ((alpha - 1.0) * obs_split.clamp_min(eps).log()).sum(dim=1)
            )
            nll = nll - ll

        return nll

    def reconstruction_nll(
        self,
        x: torch.Tensor,
        out: Dict[str, torch.Tensor],
        *,
        likelihood: Optional[LikelihoodName] = None,
        data_kind: DataKind = "relative",
        validate_counts: bool = True,
        observation_pseudocount: float = 1e-6,
    ) -> torch.Tensor:
        likelihood = self.default_likelihood if likelihood is None else likelihood

        if likelihood == "philr_gaussian":
            return self.philr_gaussian_nll(out["obs_coords"], out["coord_mu"])

        if likelihood == "multinomial":
            if data_kind != "counts":
                raise ValueError("multinomial likelihood requires data_kind='counts'.")
            return self.multinomial_nll(
                x,
                out["leaf_prob"],
                validate_counts=validate_counts,
            )

        if likelihood == "dirichlet_multinomial":
            if data_kind != "counts":
                raise ValueError("dirichlet_multinomial likelihood requires data_kind='counts'.")
            return self.dirichlet_multinomial_nll(
                x,
                out["leaf_prob"],
                validate_counts=validate_counts,
            )

        if likelihood == "dirichlet_tree_multinomial":
            if data_kind != "counts":
                raise ValueError(
                    "dirichlet_tree_multinomial likelihood requires data_kind='counts'."
                )
            return self.dirichlet_tree_multinomial_nll(
                x,
                out["leaf_prob"],
                validate_counts=validate_counts,
            )

        if likelihood == "dirichlet_tree":
            return self.dirichlet_tree_nll(
                x,
                out["leaf_prob"],
                data_kind=data_kind,
                observation_pseudocount=observation_pseudocount,
            )

        raise ValueError(f"Unknown likelihood {likelihood!r}.")

    def loss(
        self,
        x: torch.Tensor,
        out: Optional[Dict[str, torch.Tensor]] = None,
        *,
        likelihood: Optional[LikelihoodName] = None,
        data_kind: DataKind = "relative",
        beta: float = 1.0,
        free_bits: float = 0.0,
        concentration_l2: float = 1e-4,
        validate_counts: bool = True,
        observation_pseudocount: float = 1e-6,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x, data_kind=data_kind) if out is None else out
        likelihood = self.default_likelihood if likelihood is None else likelihood

        recon = self.reconstruction_nll(
            x,
            out,
            likelihood=likelihood,
            data_kind=data_kind,
            validate_counts=validate_counts,
            observation_pseudocount=observation_pseudocount,
        )

        kl = self.kl_per_sample(
            out["mu_z"],
            out["logvar_z"],
            free_bits=free_bits,
        )

        uses_tree_concentration = likelihood in {
            "dirichlet_tree",
            "dirichlet_tree_multinomial",
        }

        reg = (
            self.concentration_regularization() * float(concentration_l2)
            if uses_tree_concentration
            else x.new_zeros(())
        )

        total = recon.mean() + float(beta) * kl.mean() + reg

        metrics = {
            "loss": total.detach(),
            "reconstruction_nll": recon.mean().detach(),
            "kl": kl.mean().detach(),
            "coord_scale_mean": self.coord_scale().mean().detach(),
            "flat_concentration": self.flat_concentration().detach(),
            "concentration_l2": reg.detach(),
        }

        if self.group_depth.numel() > 0:
            metrics["group_concentration_mean"] = self.group_concentration().mean().detach()

        return total, metrics
