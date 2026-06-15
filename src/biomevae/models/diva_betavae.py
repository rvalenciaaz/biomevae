"""DIVA wrapper for the plain (non-taxonomy) β-VAE backbone.

Same MLP encoder/decoder as :class:`biomevae.models.vae.VAE`, but the
single ``(mu, logvar)`` head is replaced by three
:class:`~biomevae.models.diva.DIVAEncoderHeads` and the decoder consumes
the concatenated latent ``z = [z_d ; z_y ; z_x]``.

Reconstruction is on log1p-counts (or any user-supplied transform of
the input) and uses MSE / MAE / Huber via
:func:`biomevae.losses.reconstruction_loss` — there is no NB / ZINB
likelihood here.  This makes ``DIVABetaVAE`` the natural baseline DIVA
model: it isolates the contribution of the domain-invariance machinery
from any phylogenetic prior.

In the meta_summary table the plain ``$\\beta$-VAE`` was 5/42 best
across studies (often within 0.01 of ``dsvae-sup``), so wrapping it in
DIVA tests whether the cross-study generalisation gain attributed to
domain adaptation also materialises in the absence of compositional or
tree-structured inductive biases.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from biomevae.models.diva import DIVAEncoderHeads, DIVALoss, DIVALossOutputs
from biomevae.models.vae import get_activation


__all__ = ["DIVABetaVAE"]


class _BetaVAEEncoderTrunk(nn.Module):
    """MLP trunk identical to :class:`biomevae.models.vae.VAE`'s encoder.

    Builds ``Linear → [LayerNorm] → activation → [Dropout]`` for each
    hidden width and exposes ``feat_dim`` so the DIVA heads can attach.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        *,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.feat_dim = int(prev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BetaVAEDecoder(nn.Module):
    """MLP decoder taking the joint latent and reconstructing input space."""

    def __init__(
        self,
        latent_dim: int,
        hidden: List[int],
        output_dim: int,
        *,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden):
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DIVABetaVAE(nn.Module):
    """β-VAE with DIVA-factorised latent.

    Architecturally identical to :class:`biomevae.models.vae.VAE` except:
      * the ``(mu, logvar)`` heads are replaced by three independent
        ``(mu_d, lv_d) / (mu_y, lv_y) / (mu_x, lv_x)`` heads;
      * the decoder consumes ``z = [z_d ; z_y ; z_x]`` of total width
        ``latent_d + latent_y + latent_x``.

    No taxonomy or tree structure — inductive bias = "plain MLP +
    domain-invariance".  Use with
    :func:`biomevae.losses.reconstruction_loss` (MSE / MAE / Huber)
    for the reconstruction term.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        n_domains: int,
        n_classes: int,
        hidden: List[int] | None = None,
        latent_d: int = 4,
        latent_y: int = 8,
        latent_x: int = 8,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
        layer_norm: bool = False,
        aux_hidden: int = 64,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]
        self.trunk = _BetaVAEEncoderTrunk(
            input_dim=input_dim,
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.heads = DIVAEncoderHeads(
            feat_dim=self.trunk.feat_dim,
            latent_d=latent_d,
            latent_y=latent_y,
            latent_x=latent_x,
        )
        self.diva = DIVALoss(
            n_domains=n_domains,
            n_classes=n_classes,
            latent_d=latent_d,
            latent_y=latent_y,
            latent_x=latent_x,
            aux_hidden=aux_hidden,
            aux_dropout=dropout,
        )
        self.decoder = _BetaVAEDecoder(
            latent_dim=self.heads.total_latent,
            hidden=list(hidden),
            output_dim=input_dim,
            dropout=dropout,
            activation=activation,
            layer_norm=layer_norm,
        )

        self.input_dim = int(input_dim)
        self.latent_d = int(latent_d)
        self.latent_y = int(latent_y)
        self.latent_x = int(latent_x)
        self.n_domains = int(n_domains)
        self.n_classes = int(n_classes)

    # ------------------------------------------------------------------
    # Encode / forward / reconstruct
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        mu_d, lv_d, mu_y, lv_y, mu_x, lv_x = self.heads(h)
        return {
            "h": h,
            "mu_d": mu_d, "lv_d": lv_d,
            "mu_y": mu_y, "lv_y": lv_y,
            "mu_x": mu_x, "lv_x": lv_x,
        }

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return DIVAEncoderHeads.reparam(mu, logvar)

    def forward(
        self,
        x: torch.Tensor,
        domain: torch.Tensor,
        klass: Optional[torch.Tensor],
        *,
        free_bits: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Full DIVA forward pass.

        Returns a dict with ``recon`` (reconstruction in input space),
        ``mu_d/y/x``/``lv_d/y/x`` (latent posterior parameters), the
        sampled latents ``z_d/y/x``/``z`` (concatenation), the input
        ``x`` (so the reconstruction NLL can read it back) and the
        :class:`~biomevae.models.diva.DIVALossOutputs` bundle.
        """
        enc = self.encode(x)
        z_d = self.reparam(enc["mu_d"], enc["lv_d"])
        z_y = self.reparam(enc["mu_y"], enc["lv_y"])
        z_x = self.reparam(enc["mu_x"], enc["lv_x"])
        z = torch.cat([z_d, z_y, z_x], dim=-1)
        recon = self.decoder(z)

        diva_out = self.diva(
            mu_d=enc["mu_d"], lv_d=enc["lv_d"], z_d=z_d,
            mu_y=enc["mu_y"], lv_y=enc["lv_y"], z_y=z_y,
            mu_x=enc["mu_x"], lv_x=enc["lv_x"],
            domain=domain, klass=klass,
            free_bits=free_bits,
        )
        return {
            **enc,
            "z_d": z_d, "z_y": z_y, "z_x": z_x, "z": z,
            "x": x, "recon": recon,
            "diva": diva_out,
        }

    def reconstruct(
        self,
        mu_d: torch.Tensor,
        mu_y: torch.Tensor,
        mu_x: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic reconstruction from the per-factor latent means."""
        z = torch.cat([mu_d, mu_y, mu_x], dim=-1)
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Helpers (mirror DIVATreeNBVAE for a uniform calling convention)
    # ------------------------------------------------------------------

    def latent_split(self, mu_combined: torch.Tensor) -> Dict[str, torch.Tensor]:
        d, y, x = self.latent_d, self.latent_y, self.latent_x
        return {
            "z_d": mu_combined[..., :d],
            "z_y": mu_combined[..., d : d + y],
            "z_x": mu_combined[..., d + y : d + y + x],
        }

    @staticmethod
    def diva_loss_combine(
        diva_out: DIVALossOutputs,
        *,
        beta: float,
        alpha_d: float,
        alpha_y: float,
        batch_size: int,
    ) -> torch.Tensor:
        kl = diva_out.kl_d + diva_out.kl_y + diva_out.kl_x
        loss = beta * kl + alpha_d * diva_out.ce_d
        if diva_out.n_y_labelled > 0:
            scale = float(diva_out.n_y_labelled) / max(1, int(batch_size))
            loss = loss + alpha_y * diva_out.ce_y * scale
        return loss
