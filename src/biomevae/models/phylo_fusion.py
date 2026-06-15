from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .vae import get_activation


class DeepPhyloFusionVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        dropout: float,
        activation: str,
        layer_norm: bool,
        phylo_embeddings: torch.Tensor,
        phylo_method: str | None = None,
        phylo_dim: int | None = None,
    ) -> None:
        super().__init__()
        if phylo_embeddings.dim() != 2:
            raise ValueError("phylo_embeddings must be a 2-D tensor [features, dim].")
        self.register_buffer("phylo_embeddings", phylo_embeddings.float())
        self.phylo_dim = int(phylo_embeddings.size(1))

        enc_layers = []
        prev = input_dim + self.phylo_dim
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

        dec_layers = []
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

    def _phylo_summary(self, weights: torch.Tensor) -> torch.Tensor:
        weights = torch.clamp(weights, min=0.0)
        denom = weights.sum(dim=1, keepdim=True)
        denom = torch.where(denom > 0.0, denom, torch.ones_like(denom))
        weights = weights / denom
        return weights @ self.phylo_embeddings

    def encode(self, x: torch.Tensor, phylo_weights: torch.Tensor | None = None):
        weights = x if phylo_weights is None else phylo_weights
        summary = self._phylo_summary(weights)
        h_in = torch.cat([x, summary], dim=1)
        h = self.encoder(h_in)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, phylo_weights: torch.Tensor | None = None):
        mu, logvar = self.encode(x, phylo_weights=phylo_weights)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def prepare_fusion_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    phylo = out.get("phylo_embeddings")
    if phylo is None:
        raise ValueError("phylo_embeddings must be provided for fusion models.")
    tensor = torch.tensor(phylo, dtype=torch.float32)
    out["phylo_embeddings"] = tensor
    return out
