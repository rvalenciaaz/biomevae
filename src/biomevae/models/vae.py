from typing import List
import torch
import torch.nn as nn

#several activations functions can be tested
def get_activation(name: str):
    name = name.lower()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")

class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        latent_dim: int,
        dropout: float = 0.0,
        activation: str = "leakyrelu", # leakyrelu by default
        layer_norm: bool = False,  # layer normalisation off by default
    ):
        super().__init__()
        act_ctor = get_activation

        enc, prev = [], input_dim
        for h in hidden:
            enc.append(nn.Linear(prev, h))
            if layer_norm:  #adding layer norm if on
                enc.append(nn.LayerNorm(h))
            enc.append(act_ctor(activation))
            if dropout > 0:  #adding dropout
                enc.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

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

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
