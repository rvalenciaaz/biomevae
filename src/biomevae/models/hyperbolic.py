from typing import List
import torch
import torch.nn as nn

try:
    import geoopt
except ImportError as e:
    raise ImportError("Hyperbolic VAE requires geoopt. Install with `pip install -e .[hyper]`") from e

from .vae import get_activation

class HyperbolicVAE(nn.Module):
    """Hyperbolic VAE on the Poincaré ball (curvature -c, c>0)."""
    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        latent_dim: int,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
        layer_norm: bool = False,
        curvature: float = 1.0,
    ):
        super().__init__()
        if curvature <= 0:
            raise ValueError("curvature must be > 0 (ball curvature c).")
        # poincare ball model with negative curvature for hyperbolic space
        self.manifold = geoopt.manifolds.PoincareBallExact(c=curvature)

        act_ctor = get_activation
        enc, prev = [], input_dim
        for h in hidden:
            enc.append(nn.Linear(prev, h))
            if layer_norm:
                enc.append(nn.LayerNorm(h))
            enc.append(act_ctor(activation))
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)       # tangent @ 0
        self.fc_logvar = nn.Linear(prev, latent_dim)   # tangent @ 0

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

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    #enforcing hyperbolic structure at the reparametrisation step
    def _reparameterize_hyperbolic(self, mu_tan: torch.Tensor, logvar_tan: torch.Tensor):
        std = torch.exp(0.5 * logvar_tan)
        eps = torch.randn_like(std)
        v = mu_tan + eps * std                       # tangent sample
        z = self.manifold.expmap0(v)                 # to ball
        z = self.manifold.projx(z)                   # numeric safety
        return z

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self._reparameterize_hyperbolic(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
