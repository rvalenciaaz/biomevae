"""DEPRECATED: FlowXFormer VAE.

The FlowXFormer model is no longer recommended; use :class:`PhILRVAE` or one
of the DS-/Hyperbolic-PhILR variants instead.

The shared tree-spec utilities (:class:`TreeSpec`, :func:`build_tree_spec`,
:data:`RANK_LENGTHS`) live in :mod:`biomevae.models.tree_spec` and are
re-exported here for backwards compatibility with old configs/checkpoints.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.tree_spec import RANK_LENGTHS, TreeSpec, build_tree_spec
from biomevae.models.vae import get_activation


__all__ = [
    "RANK_LENGTHS",
    "TreeSpec",
    "build_tree_spec",
    "build_edge_distance_buckets",
    "FlowFeaturizer",
    "FlowXFormerEncoder",
    "FlowXFormerVAE",
]


def _node_distance(u: int, v: int, parent: np.ndarray, depth: np.ndarray) -> int:
    du = int(depth[u])
    dv = int(depth[v])
    uu = int(u)
    vv = int(v)
    while du > dv:
        uu = int(parent[uu])
        du -= 1
    while dv > du:
        vv = int(parent[vv])
        dv -= 1
    while uu != vv:
        uu = int(parent[uu])
        vv = int(parent[vv])
    return int(depth[u] + depth[v] - 2 * depth[uu])


def build_edge_distance_buckets(
    tree_spec: TreeSpec,
    max_bucket: int = 8,
) -> Tuple[np.ndarray, int]:
    edge_child = tree_spec.edge_child
    n_edges = len(edge_child)
    buckets = np.zeros((n_edges, n_edges), dtype=np.int64)
    for i in range(n_edges):
        for j in range(n_edges):
            dist = _node_distance(
                int(edge_child[i]),
                int(edge_child[j]),
                tree_spec.parent,
                tree_spec.node_depth,
            )
            buckets[i, j] = dist if dist <= max_bucket else max_bucket + 1
    return buckets, max_bucket + 2


class FlowFeaturizer(nn.Module):
    def __init__(
        self,
        tree_spec: TreeSpec,
        reference: np.ndarray,
        uot_mode: str = "root_l1",
        uot_lambda: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if uot_mode not in {"off", "root_l1"}:
            raise ValueError("uot_mode must be 'off' or 'root_l1'.")
        self.uot_mode = uot_mode
        self.uot_lambda = float(uot_lambda)
        self.eps = float(eps)
        self.register_buffer("parent", torch.tensor(tree_spec.parent, dtype=torch.long))
        self.register_buffer("leaf_nodes", torch.tensor(tree_spec.leaf_nodes, dtype=torch.long))
        self.register_buffer("postorder", torch.tensor(tree_spec.postorder, dtype=torch.long))
        self.register_buffer("edge_child", torch.tensor(tree_spec.edge_child, dtype=torch.long))
        self.register_buffer("edge_length", torch.tensor(tree_spec.edge_length, dtype=torch.float32))
        self.register_buffer("edge_depth", torch.tensor(tree_spec.edge_depth, dtype=torch.float32))
        self.register_buffer("reference", torch.tensor(reference, dtype=torch.float32))
        ref_scale = float(np.sqrt(reference.sum()) + eps) if uot_mode == "root_l1" else 1.0
        self.register_buffer("reference_scale", torch.tensor(ref_scale, dtype=torch.float32))

    def _leaf_mass(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.float()  # Force float32 to prevent AMP float16 overflow
        x = torch.clamp(x, min=0.0)  # Raw counts are non-negative; clamp replaces softplus
        if self.uot_mode == "root_l1":
            sums = x.sum(dim=1, keepdim=True)
            scale = torch.sqrt(sums + self.eps)
            p_leaf = x / scale
            r_leaf = self.reference / self.reference_scale
            total_mass = sums
        else:
            sums = x.sum(dim=1, keepdim=True)
            p_leaf = x / (sums + self.eps)
            r_leaf = self.reference
            total_mass = sums
        return p_leaf, r_leaf, total_mass

    def compute_flows(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        num_nodes = int(self.parent.numel())
        p_leaf, r_leaf, _total_mass = self._leaf_mass(x)
        diff = p_leaf.new_zeros((batch, num_nodes))
        diff[:, self.leaf_nodes] = p_leaf - r_leaf
        for node in self.postorder:
            node = int(node)
            parent = int(self.parent[node])
            diff[:, parent] += diff[:, node]
        flows = diff[:, self.edge_child]

        mass = p_leaf.new_zeros((batch, num_nodes))
        mass[:, self.leaf_nodes] = p_leaf
        for node in self.postorder:
            node = int(node)
            parent = int(self.parent[node])
            mass[:, parent] += mass[:, node]
        subtree_mass = mass[:, self.edge_child]
        total_mass = mass[:, 0]
        root_mismatch = diff[:, 0]
        return flows, root_mismatch, subtree_mass, total_mass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flows, root_mismatch, subtree_mass, total_mass = self.compute_flows(x)
        subtree_mass = subtree_mass.clamp(min=0.0)  # Already non-negative from tree sum
        edge_len = self.edge_length.view(1, -1)
        edge_depth = self.edge_depth.view(1, -1)
        tokens = torch.stack(
            [
                torch.asinh(flows),
                torch.asinh(flows.abs()),
                edge_len.expand_as(flows),
                edge_depth.expand_as(flows),
                torch.log1p(subtree_mass),
            ],
            dim=2,
        )
        if self.uot_mode == "root_l1":
            root_mismatch = root_mismatch.unsqueeze(1)
            total_mass = total_mass.clamp(min=0.0).unsqueeze(1)  # Already non-negative
            scaled_root = self.uot_lambda * root_mismatch
            virtual = torch.stack(
                [
                    torch.asinh(scaled_root),
                    torch.asinh(scaled_root.abs()),
                    torch.zeros_like(root_mismatch),
                    torch.zeros_like(root_mismatch),
                    torch.log1p(total_mass),
                ],
                dim=2,
            )
            tokens = torch.cat([tokens, virtual], dim=1)
        return tokens, flows, root_mismatch


class BiasTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        dim_ff: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class FlowXFormerEncoder(nn.Module):
    def __init__(
        self,
        tree_spec: TreeSpec,
        token_dim: int = 5,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        distance_bucket_max: int = 8,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        buckets, n_buckets = build_edge_distance_buckets(tree_spec, max_bucket=distance_bucket_max)
        self.register_buffer(
            "edge_distance_buckets",
            torch.tensor(buckets, dtype=torch.int16),
        )
        self.edge_bias = nn.Embedding(n_buckets, 1)
        self.virtual_bias = nn.Parameter(torch.zeros(1))
        self.input_proj = nn.Linear(token_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList(
            [
                BiasTransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    dim_ff=d_model * 4,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def _build_attn_bias(
        self,
        has_virtual: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buckets = self.edge_distance_buckets.to(device=device, dtype=torch.int32)
        base = self.edge_bias(buckets).squeeze(-1).to(dtype=dtype)
        if has_virtual:
            n_edges = base.size(0)
            bias = base.new_zeros((n_edges + 1, n_edges + 1), dtype=dtype)
            bias[:n_edges, :n_edges] = base
            virtual_bias = self.virtual_bias.to(device=device, dtype=dtype)
            bias[n_edges, :n_edges] = virtual_bias
            bias[:n_edges, n_edges] = virtual_bias
        else:
            bias = base
        full = bias.new_zeros((bias.size(0) + 1, bias.size(1) + 1), dtype=dtype)
        full[1:, 1:] = bias
        return full

    def forward(self, tokens: torch.Tensor, has_virtual: bool) -> torch.Tensor:
        batch = tokens.size(0)
        x = self.input_proj(tokens)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        attn_bias = self._build_attn_bias(has_virtual, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, attn_bias)
        x = self.norm(x)
        return x[:, 0]


class FlowXFormerVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        latent_dim: int,
        tree_spec: TreeSpec,
        reference: np.ndarray,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "leakyrelu",
        layer_norm: bool = False,
        uot_mode: str = "root_l1",
        uot_lambda: float = 0.1,
        distance_bucket_max: int = 8,
    ) -> None:
        super().__init__()
        self.featurizer = FlowFeaturizer(
            tree_spec=tree_spec,
            reference=reference,
            uot_mode=uot_mode,
            uot_lambda=uot_lambda,
        )
        self.encoder = FlowXFormerEncoder(
            tree_spec=tree_spec,
            token_dim=5,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            distance_bucket_max=distance_bucket_max,
        )
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        act_ctor = get_activation
        dec, prev = [], latent_dim
        for h in reversed(hidden):
            dec.append(nn.Linear(prev, h))
            if layer_norm:
                dec.append(nn.LayerNorm(h))
            dec.append(act_ctor(activation))
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)
        self.uot_mode = uot_mode

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, _, _ = self.featurizer(x)
        h = self.encoder(tokens, has_virtual=self.uot_mode == "root_l1")
        mu = self.fc_mu(h).float()
        logvar = torch.clamp(self.fc_logvar(h).float(), min=-20.0, max=20.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def flow_vector(self, x: torch.Tensor) -> torch.Tensor:
        flows, _, _, _ = self.featurizer.compute_flows(x)
        return torch.asinh(flows)
