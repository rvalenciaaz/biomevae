"""Tree-structured VAE for microbiome compositions/counts.

This replaces the statistically problematic "TreeNB-VAE" framing with two
likelihoods that respect the tree and the compositional sample space:

* ``dirichlet_tree_multinomial`` for true non-negative integer counts.
* ``dirichlet_tree`` for relative abundance / closed compositions.

The decoder predicts local sibling split probabilities at every internal node.
The likelihood is a product over internal nodes, so dispersion is learned at the
clade/split level rather than as unrelated per-leaf NB dispersion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.taxonomy_tree import (
    AlignmentReport,
    TaxonomyGraph,
    aggregate_leaf_matrix_to_nodes,
    align_table_to_tree_leaves,
    build_taxonomy_graph_from_phyla_tsv,
    close_composition,
    load_feature_table_as_samples_by_feature,
    validate_nonnegative_integer_counts,
)


LikelihoodName = Literal["dirichlet_tree_multinomial", "tree_multinomial", "dirichlet_tree"]


@dataclass
class TreeTopology:
    n_nodes: int
    n_leaves: int
    n_edges: int
    n_groups: int
    max_children: int
    root_id: int
    edge_parent: np.ndarray
    edge_child: np.ndarray
    edge_to_group: np.ndarray
    sibling_ranges: List[Tuple[int, int]]
    group_parent: np.ndarray
    group_depth: np.ndarray
    group_child_nodes: np.ndarray
    group_child_mask: np.ndarray
    depth_edge_indices: np.ndarray
    depth_boundaries: np.ndarray
    leaf_node_ids: np.ndarray


def build_tree_topology(taxg: TaxonomyGraph, *, include_unary_groups: bool = True) -> TreeTopology:
    """Build decoder/likelihood topology from a :class:`TaxonomyGraph`."""
    all_nodes = set(range(len(taxg.node_names)))
    child_nodes = set(taxg.parent_of.keys())
    roots = sorted(all_nodes - child_nodes)
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root; found {roots}.")
    root_id = roots[0]

    edges: List[Tuple[int, int]] = []
    for parent, children in taxg.children_of.items():
        if not include_unary_groups and len(children) == 1:
            pass
        for child in children:
            edges.append((parent, child))
    edges.sort(key=lambda x: (x[0], x[1]))

    edge_parent = np.asarray([p for p, _ in edges], dtype=np.int64)
    edge_child = np.asarray([c for _, c in edges], dtype=np.int64)
    n_edges = len(edges)
    if n_edges == 0:
        raise ValueError("Tree has no edges.")

    sibling_ranges: List[Tuple[int, int]] = []
    group_parent: List[int] = []
    edge_to_group = np.empty(n_edges, dtype=np.int64)
    start = 0
    group_id = 0
    for i in range(1, n_edges + 1):
        if i == n_edges or edge_parent[i] != edge_parent[start]:
            sibling_ranges.append((start, i))
            group_parent.append(int(edge_parent[start]))
            edge_to_group[start:i] = group_id
            group_id += 1
            start = i

    n_groups = len(sibling_ranges)
    max_children = max(e - s for s, e in sibling_ranges)
    group_child_nodes = np.full((n_groups, max_children), -1, dtype=np.int64)
    group_child_mask = np.zeros((n_groups, max_children), dtype=bool)
    for g, (s, e) in enumerate(sibling_ranges):
        k = e - s
        group_child_nodes[g, :k] = edge_child[s:e]
        group_child_mask[g, :k] = True

    node_depth = taxg.node_depth.numpy() if hasattr(taxg, "node_depth") else taxg.node_rank.numpy()
    group_depth = np.asarray([int(node_depth[p]) for p in group_parent], dtype=np.int64)

    child_depth = node_depth[edge_child]
    depth_order = np.argsort(child_depth, kind="stable").astype(np.int64)
    sorted_depths = child_depth[depth_order]
    boundaries = [0]
    for i in range(1, n_edges):
        if sorted_depths[i] != sorted_depths[i - 1]:
            boundaries.append(i)
    boundaries.append(n_edges)

    return TreeTopology(
        n_nodes=len(taxg.node_names),
        n_leaves=len(taxg.leaf_ids),
        n_edges=n_edges,
        n_groups=n_groups,
        max_children=max_children,
        root_id=root_id,
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_to_group=edge_to_group,
        sibling_ranges=sibling_ranges,
        group_parent=np.asarray(group_parent, dtype=np.int64),
        group_depth=group_depth,
        group_child_nodes=group_child_nodes,
        group_child_mask=group_child_mask,
        depth_edge_indices=depth_order,
        depth_boundaries=np.asarray(boundaries, dtype=np.int64),
        leaf_node_ids=np.asarray(taxg.leaf_ids, dtype=np.int64),
    )


def _inverse_softplus(x: float) -> float:
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class TreeBalanceEncoder(nn.Module):
    """Encoder using tree-local, sibling-centered log-ratio features."""

    def __init__(
        self,
        topo: TreeTopology,
        hidden: int = 256,
        latent_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        pseudocount: float = 0.5,
    ) -> None:
        super().__init__()
        self.pseudocount = float(pseudocount)
        self.input_dim = topo.n_edges + 1
        self.n_groups = topo.n_groups

        self.register_buffer("edge_child", torch.as_tensor(topo.edge_child, dtype=torch.long))
        self.register_buffer("edge_to_group", torch.as_tensor(topo.edge_to_group, dtype=torch.long))
        self.register_buffer(
            "group_sizes",
            torch.as_tensor([e - s for s, e in topo.sibling_ranges], dtype=torch.float32),
        )
        self.register_buffer("root_id", torch.tensor(topo.root_id, dtype=torch.long))

        layers: List[nn.Module] = []
        prev = self.input_dim
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def make_features(self, node_values: torch.Tensor) -> torch.Tensor:
        node_values = node_values.clamp_min(0.0)
        child_values = node_values.index_select(1, self.edge_child)
        log_child = torch.log(child_values + self.pseudocount)

        batch = node_values.size(0)
        group_sum = log_child.new_zeros(batch, self.n_groups)
        group_index = self.edge_to_group.unsqueeze(0).expand(batch, -1)
        group_sum.scatter_add_(1, group_index, log_child)
        group_mean = group_sum / self.group_sizes.unsqueeze(0).clamp_min(1.0)
        centered = log_child - group_mean.gather(1, group_index)

        total = node_values[:, int(self.root_id)].unsqueeze(1)
        log_total = torch.log1p(total)
        return torch.cat([centered, log_total], dim=1)

    def forward(self, node_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(self.make_features(node_values))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)


class GroupedTreeSoftmax(nn.Module):
    """Map edge logits to local split probabilities and leaf probabilities."""

    def __init__(self, topo: TreeTopology) -> None:
        super().__init__()
        pad_idx = torch.empty(topo.n_edges, dtype=torch.long)
        for g, (s, e) in enumerate(topo.sibling_ranges):
            for pos, ei in enumerate(range(s, e)):
                pad_idx[ei] = g * topo.max_children + pos

        self.register_buffer("pad_idx", pad_idx)
        self.register_buffer("edge_parent", torch.as_tensor(topo.edge_parent, dtype=torch.long))
        self.register_buffer("edge_child", torch.as_tensor(topo.edge_child, dtype=torch.long))
        self.register_buffer("depth_edge_indices", torch.as_tensor(topo.depth_edge_indices, dtype=torch.long))
        self.register_buffer("depth_boundaries", torch.as_tensor(topo.depth_boundaries, dtype=torch.long))
        self.register_buffer("leaf_node_ids", torch.as_tensor(topo.leaf_node_ids, dtype=torch.long))

        self.root_id = int(topo.root_id)
        self.n_nodes = int(topo.n_nodes)
        self.n_edges = int(topo.n_edges)
        self.n_groups = int(topo.n_groups)
        self.max_children = int(topo.max_children)
        self.sibling_ranges = topo.sibling_ranges

        # Precompute per-depth slices on CPU once so the forward pass does
        # not synchronise the GPU by repeatedly calling ``.tolist()`` inside
        # the depth loop.  ``_depth_slices`` is a Python list of (start,
        # stop) tuples in the order parents must be visited.
        boundaries = topo.depth_boundaries.tolist()
        self._depth_slices: List[Tuple[int, int]] = [
            (int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(len(boundaries) - 1)
        ]

    def forward(self, edge_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = edge_logits.size(0)
        idx = self.pad_idx.unsqueeze(0).expand(batch, -1)

        padded = edge_logits.new_full((batch, self.n_groups * self.max_children), float("-inf"))
        padded.scatter_(1, idx, edge_logits)
        padded = padded.view(batch, self.n_groups, self.max_children)

        local_log = F.log_softmax(padded, dim=2).reshape(batch, -1).gather(1, idx)

        node_log = edge_logits.new_full((batch, self.n_nodes), float("-inf"))
        node_log[:, self.root_id] = 0.0

        # Walk the tree depth-by-depth.  The number of iterations equals the
        # number of distinct child depths (typically 6-8) so the cost is
        # dominated by the per-depth vector ops, not Python overhead.
        for b0, b1 in self._depth_slices:
            eis = self.depth_edge_indices[b0:b1]
            parents = self.edge_parent.index_select(0, eis)
            children = self.edge_child.index_select(0, eis)
            update = node_log.index_select(1, parents) + local_log.index_select(1, eis)
            node_log = node_log.index_copy(1, children, update)

        leaf_log = node_log.index_select(1, self.leaf_node_ids)
        leaf_prob = F.softmax(leaf_log, dim=1)

        return {
            "edge_log_prob": local_log,
            "node_log_prob": node_log,
            "leaf_log_prob": leaf_log,
            "leaf_prob": leaf_prob,
        }


class TreeDTMDecoder(nn.Module):
    """Latent-to-tree decoder with rank-shrunk node concentrations."""

    def __init__(
        self,
        topo: TreeTopology,
        latent_dim: int = 32,
        hidden: int = 256,
        n_layers: int = 2,
        init_concentration: float = 50.0,
        min_concentration: float = 1e-3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = latent_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(prev, hidden), nn.SiLU()])
            prev = hidden
        layers.append(nn.Linear(prev, topo.n_edges))

        self.edge_mlp = nn.Sequential(*layers)
        self.tree_softmax = GroupedTreeSoftmax(topo)
        self.min_concentration = float(min_concentration)

        max_depth = int(np.max(topo.group_depth)) if topo.n_groups else 0
        raw = _inverse_softplus(float(init_concentration))
        self.rank_raw_concentration = nn.Parameter(torch.full((max_depth + 1,), raw))
        self.group_raw_delta = nn.Parameter(torch.zeros(topo.n_groups))
        self.register_buffer("group_depth", torch.as_tensor(topo.group_depth, dtype=torch.long))

    def group_concentration(self) -> torch.Tensor:
        raw = self.rank_raw_concentration.index_select(0, self.group_depth) + self.group_raw_delta
        return F.softplus(raw) + self.min_concentration

    def concentration_regularization(self) -> torch.Tensor:
        return self.group_raw_delta.pow(2).mean()

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        edge_logits = self.edge_mlp(z)
        out = self.tree_softmax(edge_logits)
        out["edge_logits"] = edge_logits
        out["group_concentration"] = self.group_concentration()
        return out


class TreeDTMVAE(nn.Module):
    """Tree VAE for counts or relative abundances.

    Parameters
    ----------
    likelihood:
        ``dirichlet_tree_multinomial`` for true counts; ``tree_multinomial`` for
        counts without overdispersion; ``dirichlet_tree`` for relative abundance
        compositions.
    """

    def __init__(
        self,
        topo: TreeTopology,
        *,
        hidden: int = 256,
        latent_dim: int = 32,
        encoder_layers: int = 2,
        decoder_hidden: int = 256,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        encoder_pseudocount: float = 0.5,
        init_concentration: float = 50.0,
        likelihood: LikelihoodName = "dirichlet_tree_multinomial",
    ) -> None:
        super().__init__()
        self.topo = topo
        self.default_likelihood: LikelihoodName = likelihood

        self.encoder = TreeBalanceEncoder(
            topo,
            hidden=hidden,
            latent_dim=latent_dim,
            n_layers=encoder_layers,
            dropout=dropout,
            pseudocount=encoder_pseudocount,
        )
        self.decoder = TreeDTMDecoder(
            topo,
            latent_dim=latent_dim,
            hidden=decoder_hidden,
            n_layers=decoder_layers,
            init_concentration=init_concentration,
        )

        self.register_buffer("group_child_nodes", torch.as_tensor(topo.group_child_nodes, dtype=torch.long))
        self.register_buffer("group_child_mask", torch.as_tensor(topo.group_child_mask, dtype=torch.bool))
        # Flat buffers used by the vectorised NLL implementations. They let
        # us replace the per-group Python loops (~``n_groups`` iterations
        # per forward, often >10k) with a handful of fused tensor ops.
        self.register_buffer("edge_child_flat", torch.as_tensor(topo.edge_child, dtype=torch.long))
        self.register_buffer("edge_to_group_flat", torch.as_tensor(topo.edge_to_group, dtype=torch.long))
        self.register_buffer(
            "group_parent_node",
            torch.as_tensor(topo.group_parent, dtype=torch.long),
        )
        self.register_buffer(
            "group_size_float",
            torch.as_tensor(
                [e - s for s, e in topo.sibling_ranges], dtype=torch.float32
            ),
        )
        self.sibling_ranges = topo.sibling_ranges
        self._n_groups_int = int(topo.n_groups)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
        logvar = logvar.clamp(min=-30.0, max=20.0)
        per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        if free_bits > 0.0:
            per_dim = torch.clamp(per_dim, min=float(free_bits))
        return per_dim.sum(dim=1)

    def encode(self, node_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(node_values)

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(z)

    def forward(self, node_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu_z, logvar_z = self.encode(node_values)
        z = self.reparam(mu_z, logvar_z)
        out = self.decode(z)
        out.update({"mu_z": mu_z, "logvar_z": logvar_z, "z": z, "y_nodes": node_values})
        return out

    @torch.no_grad()
    def reconstruct_leaf_proportions(self, node_values: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        mu_z, logvar_z = self.encode(node_values)
        z = mu_z if use_mean else self.reparam(mu_z, logvar_z)
        return self.decode(z)["leaf_prob"]

    def _check_count_tensor(self, node_values: torch.Tensor) -> None:
        if not torch.isfinite(node_values).all() or (node_values < 0).any():
            raise ValueError("node_values must be finite and non-negative.")
        if not torch.allclose(node_values, node_values.round(), atol=1e-4, rtol=0.0):
            raise ValueError("Count likelihood requested, but node_values contain non-integer values.")

    # ------------------------------------------------------------------
    # Vectorised tree likelihoods.
    #
    # The original implementations walked every sibling group in a Python
    # ``for`` loop and called ``index_select`` / ``sum`` for each group.
    # On real trees that can mean 10k+ Python iterations per training step
    # — utterly dominating the cost.  The implementations below rewrite
    # the same maths as a handful of fused tensor ops over the flat
    # ``(batch, n_edges)`` and ``(batch, n_groups)`` axes.
    #
    # Notation used inside each method (B = batch size):
    #   ``x_edge``      : (B, n_edges)   – child-node counts/proportions.
    #   ``n_group``     : (B, n_groups)  – parent-node totals (the count
    #                                       at the group's parent node).
    #   ``alpha``       : (B, n_edges)   – per-edge Dirichlet parameter.
    #   ``alpha0_group``: (B, n_groups)  – per-group ``alpha`` sum.
    # The group totals are read directly from ``node_values`` at the
    # parent-node index for that group; ``aggregate_leaf_matrix_to_nodes``
    # already stored the descendant sum at every internal node, so we do
    # not need to re-sum children here.
    # ------------------------------------------------------------------

    def _scatter_sum_to_groups(self, per_edge: torch.Tensor) -> torch.Tensor:
        """Sum a ``(B, n_edges)`` tensor down to ``(B, n_groups)``."""
        batch = per_edge.size(0)
        idx = self.edge_to_group_flat.unsqueeze(0).expand(batch, -1)
        out = per_edge.new_zeros(batch, self._n_groups_int)
        out.scatter_add_(1, idx, per_edge)
        return out

    def tree_multinomial_nll(
        self,
        node_values: torch.Tensor,
        edge_log_prob: torch.Tensor,
        *,
        validate_counts: bool = True,
    ) -> torch.Tensor:
        if validate_counts:
            self._check_count_tensor(node_values)

        x_edge = node_values.index_select(1, self.edge_child_flat)
        n_group = node_values.index_select(1, self.group_parent_node)

        ll = (
            torch.lgamma(n_group + 1.0).sum(dim=1)
            - torch.lgamma(x_edge + 1.0).sum(dim=1)
            + (x_edge * edge_log_prob).sum(dim=1)
        )
        return -ll

    def dirichlet_tree_multinomial_nll(
        self,
        node_values: torch.Tensor,
        edge_log_prob: torch.Tensor,
        group_concentration: torch.Tensor,
        *,
        validate_counts: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if validate_counts:
            self._check_count_tensor(node_values)

        x_edge = node_values.index_select(1, self.edge_child_flat)
        n_group = node_values.index_select(1, self.group_parent_node)

        conc_per_edge = group_concentration.clamp_min(eps).index_select(
            0, self.edge_to_group_flat
        )
        p_edge = edge_log_prob.exp().clamp_min(eps)
        alpha = p_edge * conc_per_edge.unsqueeze(0) + eps

        alpha0_group = self._scatter_sum_to_groups(alpha)

        ll = (
            torch.lgamma(n_group + 1.0).sum(dim=1)
            - torch.lgamma(x_edge + 1.0).sum(dim=1)
            + torch.lgamma(alpha0_group).sum(dim=1)
            - torch.lgamma(n_group + alpha0_group).sum(dim=1)
            + torch.lgamma(x_edge + alpha).sum(dim=1)
            - torch.lgamma(alpha).sum(dim=1)
        )
        return -ll

    def dirichlet_tree_nll(
        self,
        node_values: torch.Tensor,
        edge_log_prob: torch.Tensor,
        group_concentration: torch.Tensor,
        *,
        observation_pseudocount: float = 1e-6,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Continuous Dirichlet-tree density for closed relative abundances.

        ``node_values`` should be non-negative and usually have root value 1. The
        observed child proportions at each internal node are smoothed only to keep
        zeros inside the Dirichlet support.
        """
        if not torch.isfinite(node_values).all() or (node_values < 0).any():
            raise ValueError("node_values must be finite and non-negative.")

        obs_pc = float(observation_pseudocount)
        batch = node_values.size(0)

        x_edge = node_values.index_select(1, self.edge_child_flat)
        n_group = node_values.index_select(1, self.group_parent_node)

        # Broadcast per-group totals back out to per-edge so that the
        # smoothed proportion at every edge can be computed in one shot.
        edge_to_group = self.edge_to_group_flat
        edge_to_group_b = edge_to_group.unsqueeze(0).expand(batch, -1)
        n_per_edge = n_group.gather(1, edge_to_group_b)
        k_per_edge = self.group_size_float.index_select(0, edge_to_group).unsqueeze(0)

        prop_edge = (x_edge + obs_pc) / (n_per_edge + obs_pc * k_per_edge)
        log_prop_edge = prop_edge.clamp_min(eps).log()

        conc_per_edge = group_concentration.clamp_min(eps).index_select(0, edge_to_group)
        p_edge = edge_log_prob.exp().clamp_min(eps)
        alpha = p_edge * conc_per_edge.unsqueeze(0) + eps

        alpha0_group = self._scatter_sum_to_groups(alpha)
        lgamma_alpha_group = self._scatter_sum_to_groups(torch.lgamma(alpha))
        inner_group = self._scatter_sum_to_groups((alpha - 1.0) * log_prop_edge)

        ll_per_group = torch.lgamma(alpha0_group) - lgamma_alpha_group + inner_group

        # The original implementation zero-masked groups whose observed
        # total was non-positive (they contribute no information). Keep
        # the same behaviour here.
        valid_group = n_group > 0
        ll_per_group = torch.where(valid_group, ll_per_group, torch.zeros_like(ll_per_group))

        return -ll_per_group.sum(dim=1)

    def loss(
        self,
        node_values: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        *,
        likelihood: Optional[LikelihoodName] = None,
        beta: float = 1.0,
        free_bits: float = 0.0,
        concentration_l2: float = 1e-4,
        validate_counts: bool = True,
        observation_pseudocount: float = 1e-6,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.forward(node_values) if outputs is None else outputs
        likelihood = self.default_likelihood if likelihood is None else likelihood

        edge_log_prob = outputs["edge_log_prob"]
        group_conc = outputs["group_concentration"]

        if likelihood == "tree_multinomial":
            recon = self.tree_multinomial_nll(node_values, edge_log_prob, validate_counts=validate_counts)
        elif likelihood == "dirichlet_tree_multinomial":
            recon = self.dirichlet_tree_multinomial_nll(
                node_values,
                edge_log_prob,
                group_conc,
                validate_counts=validate_counts,
            )
        elif likelihood == "dirichlet_tree":
            recon = self.dirichlet_tree_nll(
                node_values,
                edge_log_prob,
                group_conc,
                observation_pseudocount=observation_pseudocount,
            )
        else:
            raise ValueError(f"Unknown likelihood {likelihood!r}.")

        kl = self.kl_per_sample(outputs["mu_z"], outputs["logvar_z"], free_bits=free_bits)
        reg = self.decoder.concentration_regularization() * float(concentration_l2)

        loss = recon.mean() + float(beta) * kl.mean() + reg
        metrics = {
            "loss": loss.detach(),
            "reconstruction_nll": recon.mean().detach(),
            "kl": kl.mean().detach(),
            "concentration_l2": reg.detach(),
            "mean_concentration": group_conc.mean().detach(),
        }
        return loss, metrics


def build_treevae_dataset(
    sgb_table_tsv: Union[str, Path],
    phyla_tsv: Union[str, Path],
    *,
    data_kind: Literal["counts", "relative"] = "relative",
    keep_prefixes: bool = False,
    strict_alignment: bool = True,
    allow_missing_leaves: bool = False,
    allow_extra_features: bool = True,
    min_matched_fraction: float = 0.95,
    taxonomy_has_header: bool = False,
) -> Tuple[TaxonomyGraph, TreeTopology, torch.Tensor, torch.Tensor, List[str], List[str], AlignmentReport]:
    """Build tensors for TreeDTMVAE.

    Returns
    -------
    taxg, topo, X_nodes, X_leaves, sample_ids, leaf_names, alignment_report

    ``X_nodes`` is samples x all tree nodes and is the tensor consumed by the
    model. ``X_leaves`` is samples x leaves in the same leaf order as ``topo``.
    """
    taxg = build_taxonomy_graph_from_phyla_tsv(
        phyla_tsv,
        keep_prefixes=keep_prefixes,
        has_header=taxonomy_has_header,
        fill_missing_intermediate=True,
        on_duplicate_leaf="ignore_same",
    )
    topo = build_tree_topology(taxg)

    Xdf, sample_ids, _feature_ids = load_feature_table_as_samples_by_feature(sgb_table_tsv)
    X_leaf, leaf_names, report = align_table_to_tree_leaves(
        Xdf,
        taxg,
        strict=strict_alignment,
        allow_missing_leaves=allow_missing_leaves,
        allow_extra_features=allow_extra_features,
        min_matched_fraction=min_matched_fraction,
    )

    if data_kind == "counts":
        X_leaf = validate_nonnegative_integer_counts(X_leaf)
    elif data_kind == "relative":
        X_leaf = close_composition(X_leaf)
    else:
        raise ValueError("data_kind must be 'counts' or 'relative'.")

    X_nodes = aggregate_leaf_matrix_to_nodes(taxg, X_leaf)

    return (
        taxg,
        topo,
        torch.from_numpy(X_nodes.astype(np.float32)),
        torch.from_numpy(X_leaf.astype(np.float32)),
        sample_ids,
        leaf_names,
        report,
    )
