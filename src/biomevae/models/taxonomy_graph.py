from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from .vae import get_activation


def _build_normalized_adjacency(num_nodes: int, edges: Sequence[Sequence[float]]) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for edge in edges:
        if len(edge) < 3:
            raise ValueError("Each edge must be a triplet (src, dst, weight).")
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes:
            raise ValueError("Edge index out of range for taxonomy graph.")
        adj[i, j] += w
        adj[j, i] += w
    adj = adj + torch.eye(num_nodes, dtype=torch.float32)
    deg = adj.sum(dim=1)
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = torch.pow(deg, -0.5)
    norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
    return norm


class GraphPropagationLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, features: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = norm_adj @ features
        h = self.linear(h)
        h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class TaxonomyGraphEncoder(nn.Module):
    """Encode taxonomy nodes with a lightweight graph neural network."""

    def __init__(
        self,
        graph_spec: Dict[str, Any],
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        num_nodes = int(graph_spec["num_nodes"])
        edges: Sequence[Sequence[float]] = graph_spec.get("edges", [])
        feature_indices: Iterable[int] = graph_spec.get("feature_indices", [])

        if not feature_indices:
            raise ValueError("graph_spec must include feature_indices for sample pooling.")

        self.register_buffer("norm_adj", _build_normalized_adjacency(num_nodes, edges))
        self.register_buffer("node_eye", torch.eye(num_nodes, dtype=torch.float32))
        selector = torch.tensor(list(feature_indices), dtype=torch.long)
        self.register_buffer("feature_selector", selector)

        dims: List[int] = [num_nodes]
        if hidden_dims:
            dims.extend(int(h) for h in hidden_dims)
        else:
            dims.append(num_nodes)
        layers: List[GraphPropagationLayer] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(GraphPropagationLayer(in_dim, out_dim, activation, dropout))
        self.layers = nn.ModuleList(layers)
        self.output_dim = dims[-1]

    def compute_node_embeddings(self) -> torch.Tensor:
        h = self.node_eye
        for layer in self.layers:
            h = layer(h, self.norm_adj)
        return h

    def feature_embeddings(self) -> torch.Tensor:
        emb = self.compute_node_embeddings()
        return emb.index_select(0, self.feature_selector)

    @staticmethod
    def _prepare_weights(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        min_vals = x.min(dim=1, keepdim=True).values
        weights = torch.where(min_vals < 0.0, x - min_vals, x)
        weights = torch.clamp(weights, min=0.0)
        denom = weights.sum(dim=1, keepdim=True)
        denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
        normalized = weights / denom_safe
        fallback = torch.full_like(weights, 1.0 / weights.size(1))
        normalized = torch.where(denom > 0.0, normalized, fallback)
        return weights, denom, normalized

    @staticmethod
    def normalize_weights(x: torch.Tensor) -> torch.Tensor:
        _, _, normalized = TaxonomyGraphEncoder._prepare_weights(x)
        return normalized

    def sample_representation(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_embeddings()
        _, _, normalized = self._prepare_weights(x)
        pooled = normalized @ feats
        return pooled
