from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from .taxonomy_graph import TaxonomyGraphEncoder
from .vae import get_activation


def _ensure_hidden_list(hidden: Sequence[int] | int | None) -> List[int]:
    if hidden is None:
        return []
    if isinstance(hidden, int):
        return [hidden]
    return [int(h) for h in hidden]


class TaxonomyGraphVAE(nn.Module):
    """VAE encoder that pools information over a taxonomy graph."""

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        dropout: float,
        activation: str,
        layer_norm: bool,
        graph_spec: Dict[str, Any],
        gnn_hidden: Sequence[int],
        gnn_dropout: float = 0.0,
        graph_mode: str | None = None,
        gnn_type: str | None = None,
    ) -> None:
        super().__init__()

        self.graph_encoder = TaxonomyGraphEncoder(
            graph_spec=graph_spec,
            hidden_dims=_ensure_hidden_list(gnn_hidden),
            activation=activation,
            dropout=gnn_dropout,
        )

        rep_dim = self.graph_encoder.output_dim
        enc_layers: List[nn.Module] = []
        prev = rep_dim
        for h in hidden:
            enc_layers.append(nn.Linear(prev, h))
            if layer_norm:
                enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(get_activation(activation))
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: List[nn.Module] = []
        prev_dec = latent_dim
        for h in reversed(list(hidden)):
            dec_layers.append(nn.Linear(prev_dec, h))
            if layer_norm:
                dec_layers.append(nn.LayerNorm(h))
            dec_layers.append(get_activation(activation))
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            prev_dec = h
        dec_layers.append(nn.Linear(prev_dec, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        pooled = self.graph_encoder.sample_representation(x)
        h = self.encoder(pooled)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def prepare_graph_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    spec = out.get("graph_spec")
    if spec is None:
        raise ValueError("graph_spec must be provided for graph-based models.")
    out["graph_spec"] = {
        "num_nodes": int(spec["num_nodes"]),
        "edges": [
            [int(edge[0]), int(edge[1]), float(edge[2])] for edge in spec.get("edges", [])
        ],
        "feature_indices": [int(i) for i in spec.get("feature_indices", [])],
        "node_labels": list(spec.get("node_labels", [])),
    }
    out["gnn_hidden"] = _ensure_hidden_list(out.get("gnn_hidden"))
    out["gnn_dropout"] = float(out.get("gnn_dropout", 0.0))
    return out
